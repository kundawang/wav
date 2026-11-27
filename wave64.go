package wav

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// Sony Wave64 (aka Sound Forge W64) container identifiers.
//
// A Wave64 GUID is a 16 byte little-endian Windows GUID. Its first 4 bytes
// happen to match the equivalent RIFF FourCC, which is why the RIFF GUID
// starts with the ASCII characters "riff". The values are the well-known
// GUIDs defined by the Sony Wave64 specification.
//
// Note the GUID byte ordering: Data1 (4 bytes), Data2 (2 bytes) and Data3
// (2 bytes) are stored little-endian, so the fmt GUID's Data2 value 0xACF3
// appears on disk as F3 AC, not AC F3. Only the trailing 8 byte Data4 block
// is kept in its structural order.
var (
	// riffWave64GUID identifies a Wave64 RIFF container.
	riffWave64GUID = [16]byte{
		0x72, 0x69, 0x66, 0x66, // "riff" (Data1, little-endian)
		0x2E, 0x91, 0xCF, 0x11, // Data2, Data3
		0xA5, 0xD6, 0x28, 0xDB, 0x04, 0xC1, 0x00, 0x00, // Data4
	}
	// waveWave64GUID identifies the WAVE form carried by the RIFF chunk.
	waveWave64GUID = [16]byte{
		0x77, 0x61, 0x76, 0x65, // "wave"
		0xF3, 0xAC, 0xD3, 0x11,
		0x8C, 0xD1, 0x00, 0xC0, 0x4F, 0x8E, 0xDB, 0x8A,
	}
	// fmtWave64GUID identifies the format chunk.
	fmtWave64GUID = [16]byte{
		0x66, 0x6D, 0x74, 0x20, // "fmt "
		0xF3, 0xAC, 0xD3, 0x11,
		0x8C, 0xD1, 0x00, 0xC0, 0x4F, 0x8E, 0xDB, 0x8A,
	}
	// dataWave64GUID identifies the audio data chunk.
	dataWave64GUID = [16]byte{
		0x64, 0x61, 0x74, 0x61, // "data"
		0xF3, 0xAC, 0xD3, 0x11,
		0x8C, 0xD1, 0x00, 0xC0, 0x4F, 0x8E, 0xDB, 0x8A,
	}
)

var (
	// ErrWave64UnsupportedFormat is returned when a Wave64 encoder is asked to
	// encode parameters the Wave64 PCM mode of this package does not support.
	ErrWave64UnsupportedFormat = errors.New("wav: Wave64 only supports 16 or 24 bit PCM with 1 or 2 channels")
	// ErrWave64Overflow is returned when the amount of audio data would not fit
	// in the 64 bit chunk size fields of a Wave64 file.
	ErrWave64Overflow = errors.New("wav: Wave64 chunk size overflow")
	// ErrWave64Metadata is returned when metadata is attached to a Wave64
	// encoder, as the Wave64 mode of this package only writes the RIFF, fmt and
	// data chunks.
	ErrWave64Metadata = errors.New("wav: metadata chunks are not supported in Wave64 mode")
)

const (
	// wave64GUIDSize is the size in bytes of a Wave64 GUID.
	wave64GUIDSize = 16
	// wave64ChunkHeaderSize is GUID (16 bytes) + 64 bit chunk size (8 bytes).
	wave64ChunkHeaderSize = wave64GUIDSize + 8
	// wave64Alignment is the required start/size alignment of every chunk.
	wave64Alignment = 8
	// wave64FmtPayloadSize is the size of the PCM fmt chunk payload
	// (the standard 16 byte WAVEFORMATEX body).
	wave64FmtPayloadSize = 16
	// wave64FmtChunkSize is the fmt chunk size including its own header.
	wave64FmtChunkSize = wave64ChunkHeaderSize + wave64FmtPayloadSize
)

// NewWave64Encoder creates an encoder writing a Sony Wave64 (.w64) container.
//
// Unlike NewEncoder, which keeps the classic 32 bit RIFF/WAVE layout, the
// returned encoder uses 16 byte GUIDs and 64 bit chunk sizes, allowing files
// larger than 4 GiB. Only linear PCM is supported, at 16 or 24 bits per
// sample and in mono or stereo. The classic NewEncoder output format is left
// untouched.
//
// As with NewEncoder, Close must be called for the header sizes to be
// backfilled; the underlying writer is not closed.
func NewWave64Encoder(w io.WriteSeeker, sampleRate, bitDepth, numChans int) (*Encoder, error) {
	if sampleRate <= 0 {
		return nil, fmt.Errorf("wav: invalid sample rate %d", sampleRate)
	}
	if numChans != 1 && numChans != 2 {
		return nil, ErrWave64UnsupportedFormat
	}
	switch bitDepth {
	case 16, 24:
	default:
		return nil, ErrWave64UnsupportedFormat
	}
	if uint64(sampleRate)*uint64(numChans)*uint64(bitDepth/8) > math.MaxUint32 {
		return nil, fmt.Errorf("wav: average bytes per second overflows uint32: %w", ErrWave64UnsupportedFormat)
	}

	return &Encoder{
		w:              w,
		buf:            bytesBufferFor(sampleRate, bitDepth, numChans),
		SampleRate:     sampleRate,
		BitDepth:       bitDepth,
		NumChans:       numChans,
		WavAudioFormat: 1, // WAVE_FORMAT_PCM
		wave64:         true,
	}, nil
}

// IsWave64 reports whether the encoder writes a Sony Wave64 container rather
// than a classic RIFF/WAVE file.
func (e *Encoder) IsWave64() bool {
	return e != nil && e.wave64
}

// addRaw writes a fixed-size byte sequence without any endianness conversion.
// GUIDs are stored as their on-disk byte representation.
func (e *Encoder) addRaw(src []byte) error {
	n, err := e.w.Write(src)
	e.WrittenBytes += n
	return err
}

// addLE64 writes a 64 bit little-endian value.
func (e *Encoder) addLE64(v uint64) error {
	e.WrittenBytes += 8
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	_, err := e.w.Write(b[:])
	return err
}

// patchLE64 overwrites 8 bytes at the current seek position without touching
// WrittenBytes. It is used to backfill size fields after the header bytes were
// already accounted for when their placeholders were written.
func (e *Encoder) patchLE64(v uint64) error {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	_, err := e.w.Write(b[:])
	return err
}

// wave64Padding returns the number of zero padding bytes needed for payload
// to end on an 8 byte boundary.
func wave64Padding(payloadSize uint64) int {
	if r := payloadSize % wave64Alignment; r != 0 {
		return wave64Alignment - int(r)
	}
	return 0
}

// wave64DataSize computes the data chunk size (header included, as required by
// the Wave64 specification) for the given number of PCM frames, returning
// ErrWave64Overflow if the 64 bit size field would overflow.
func wave64DataSize(frames, numChans, bitDepth int) (uint64, error) {
	if frames < 0 || numChans < 0 || bitDepth < 0 {
		return 0, fmt.Errorf("wav: negative Wave64 size component (frames=%d, channels=%d, bits=%d)", frames, numChans, bitDepth)
	}
	bytesPerFrame := uint64(numChans) * uint64(bitDepth/8)
	if bytesPerFrame != 0 && uint64(frames) > math.MaxUint64/bytesPerFrame {
		return 0, ErrWave64Overflow
	}
	payload := uint64(frames) * bytesPerFrame
	size := wave64ChunkHeaderSize + payload + uint64(wave64Padding(payload))
	if size < payload {
		return 0, ErrWave64Overflow
	}
	return size, nil
}

// writeWave64Header writes the RIFF, WAVE and fmt entries. Chunk sizes are
// placeholders backfilled by Close.
func (e *Encoder) writeWave64Header() error {
	// RIFF chunk: GUID + placeholder 64 bit size.
	if err := e.addRaw(riffWave64GUID[:]); err != nil {
		return err
	}
	if err := e.addLE64(0); err != nil {
		return fmt.Errorf("error writing Wave64 RIFF size - %w", err)
	}
	// WAVE form GUID (no size, it is not a chunk).
	if err := e.addRaw(waveWave64GUID[:]); err != nil {
		return err
	}
	// fmt chunk.
	if err := e.addRaw(fmtWave64GUID[:]); err != nil {
		return err
	}
	if err := e.addLE64(wave64FmtChunkSize); err != nil {
		return fmt.Errorf("error writing Wave64 fmt size - %w", err)
	}
	if err := e.AddLE(uint16(e.WavAudioFormat)); err != nil {
		return fmt.Errorf("error encoding the wave format - %w", err)
	}
	if err := e.AddLE(uint16(e.NumChans)); err != nil {
		return fmt.Errorf("error encoding the number of channels - %w", err)
	}
	if err := e.AddLE(uint32(e.SampleRate)); err != nil {
		return fmt.Errorf("error encoding the sample rate - %w", err)
	}
	blockAlign := e.NumChans * e.BitDepth / 8
	if err := e.AddLE(uint32(e.SampleRate * blockAlign)); err != nil {
		return fmt.Errorf("error encoding the avg bytes per sec - %w", err)
	}
	if err := e.AddLE(uint16(blockAlign)); err != nil {
		return fmt.Errorf("error encoding the block align - %w", err)
	}
	if err := e.AddLE(uint16(e.BitDepth)); err != nil {
		return fmt.Errorf("error encoding bits per sample - %w", err)
	}

	return nil
}

// startWave64PCMChunk writes the data chunk header with a placeholder size.
func (e *Encoder) startWave64PCMChunk() error {
	if err := e.addRaw(dataWave64GUID[:]); err != nil {
		return fmt.Errorf("error encoding Wave64 data GUID %w", err)
	}
	e.dataSizePos = int64(e.WrittenBytes)
	if err := e.addLE64(0); err != nil {
		return fmt.Errorf("%w when writing Wave64 data chunk size header", err)
	}
	e.pcmChunkStarted = true
	return nil
}
