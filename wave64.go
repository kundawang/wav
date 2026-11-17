package wav

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/go-audio/riff"
)

// WAVE64 (Sony Wave64) container support, read-only.
//
// A WAVE64 file replaces the RIFF 4-byte IDs and 32-bit sizes with
// 16-byte GUIDs and 64-bit sizes. All sizes include their own 24-byte
// (GUID + size) header and all chunks are aligned on 8-byte boundaries.
//
// The container layout is:
//   riff GUID (16) + file size uint64 LE (8, includes these 24 bytes)
//   wave GUID (16)
//   chunks: chunk GUID (16) + chunk size uint64 LE (8, includes the 24
//   bytes of chunk header) + data, zero padded to an 8-byte boundary.

const (
	// wave64ChunkHeaderSize is the size of a chunk header: 16-byte GUID
	// plus an 8-byte little-endian size.
	wave64ChunkHeaderSize = 24
	// wave64HeaderSize is the size of the container header: the riff
	// chunk header followed by the wave form GUID.
	wave64HeaderSize = wave64ChunkHeaderSize + 16
)

var (
	// wave64RiffGUID is the GUID identifying a WAVE64 container
	// ({66696972-912E-11CF-A5D6-28DB04C10000}, starts with "riff").
	wave64RiffGUID = [16]byte{'r', 'i', 'f', 'f', 0x2E, 0x91, 0xCF, 0x11,
		0xA5, 0xD6, 0x28, 0xDB, 0x04, 0xC1, 0x00, 0x00}
	// wave64WaveGUID is the form type GUID of a WAVE64 wave container
	// ({77617665-ACF3-11D3-8CD1-00C04F8EDB8A}, starts with "wave").
	wave64WaveGUID = [16]byte{'w', 'a', 'v', 'e', 0xF3, 0xAC, 0xD3, 0x11,
		0x8C, 0xD1, 0x00, 0xC0, 0x4F, 0x8E, 0xDB, 0x8A}
	// wave64ChunkGUIDTail is the shared tail of the chunk GUIDs derived
	// from RIFF 4-byte IDs ({xxxxxxxx-ACF3-11D3-8CD1-00C04F8EDB8A}).
	wave64ChunkGUIDTail = []byte{0xF3, 0xAC, 0xD3, 0x11,
		0x8C, 0xD1, 0x00, 0xC0, 0x4F, 0x8E, 0xDB, 0x8A}
)

// WAVE64 audio format tags.
const (
	wave64FormatPCM       = 1
	wave64FormatIEEEFloat = 3
)

// detectContainer peeks at the start of the reader to figure out if the
// content is a classic RIFF/WAVE file or a WAVE64 container. RIFF readers
// are rewound so the riff parser can process them from the start.
func (d *Decoder) detectContainer() error {
	if d.containerDetected {
		return nil
	}
	var magic [16]byte
	if _, err := io.ReadFull(d.r, magic[:]); err != nil {
		return fmt.Errorf("failed to read the container header: %w", err)
	}
	if magic == wave64RiffGUID {
		d.containerDetected = true
		d.isWave64 = true
		return d.readWave64ContainerHeader()
	}
	// Not WAVE64, rewind so the RIFF parser can read the file from the
	// beginning.
	if _, err := d.r.Seek(-16, io.SeekCurrent); err != nil {
		return fmt.Errorf("failed to rewind after format detection, a seekable reader is required: %w", err)
	}
	d.containerDetected = true
	return nil
}

// readWave64ContainerHeader reads the file size and the wave form GUID
// following the riff GUID.
func (d *Decoder) readWave64ContainerHeader() error {
	var size uint64
	if err := binary.Read(d.r, binary.LittleEndian, &size); err != nil {
		return fmt.Errorf("wave64: failed to read the container size: %w", err)
	}
	if size < wave64HeaderSize {
		return fmt.Errorf("wave64: invalid container size %d, smaller than the %d bytes header", size, wave64HeaderSize)
	}
	var waveGUID [16]byte
	if _, err := io.ReadFull(d.r, waveGUID[:]); err != nil {
		return fmt.Errorf("wave64: failed to read the wave form GUID: %w", err)
	}
	if waveGUID != wave64WaveGUID {
		return fmt.Errorf("wave64: unexpected form GUID %x, not a WAVE container", waveGUID)
	}
	d.w64Size = size
	d.w64NextChunkOff = wave64HeaderSize
	// Mirror the container info into the riff parser so String() and
	// Duration() keep working like they do for RIFF files.
	d.parser.ID = riff.RiffID
	d.parser.Format = riff.WavFormatID
	if size > math.MaxUint32 {
		d.parser.Size = math.MaxUint32
	} else {
		d.parser.Size = uint32(size)
	}
	return nil
}

// wave64ChunkID maps a WAVE64 chunk GUID back to its RIFF 4-byte ID.
// Unknown GUIDs return a zero ID and are meant to be skipped by the caller.
func wave64ChunkID(guid [16]byte) [4]byte {
	var id [4]byte
	if bytes.Equal(guid[4:], wave64ChunkGUIDTail) {
		copy(id[:], guid[:4])
	}
	return id
}

// nextWave64Chunk returns the next chunk of a WAVE64 container. Chunk
// headers carry a 16-byte GUID and a 64-bit size that includes the 24
// bytes of the header itself; chunk data is padded to an 8-byte boundary.
// Unknown chunks are returned with a zero ID so the caller can drain them.
func (d *Decoder) nextWave64Chunk() (*riff.Chunk, error) {
	off := d.w64NextChunkOff
	if off < 0 || uint64(off)+wave64ChunkHeaderSize > d.w64Size {
		return nil, io.EOF
	}
	if _, err := d.r.Seek(off, io.SeekStart); err != nil {
		return nil, fmt.Errorf("wave64: failed to seek to the chunk at offset %d, a seekable reader is required: %w", off, err)
	}
	var guid [16]byte
	if _, err := io.ReadFull(d.r, guid[:]); err != nil {
		return nil, fmt.Errorf("wave64: short read on the chunk GUID at offset %d: %w", off, err)
	}
	var size uint64
	if err := binary.Read(d.r, binary.LittleEndian, &size); err != nil {
		return nil, fmt.Errorf("wave64: short read on the chunk size at offset %d: %w", off, err)
	}
	if size < wave64ChunkHeaderSize {
		return nil, fmt.Errorf("wave64: invalid chunk size %d at offset %d, smaller than the %d bytes chunk header", size, off, wave64ChunkHeaderSize)
	}
	dataSize := size - wave64ChunkHeaderSize
	// chunks are aligned on 8-byte boundaries
	padded := (size + 7) &^ 7
	if padded > d.w64Size-uint64(off) {
		return nil, fmt.Errorf("wave64: chunk at offset %d with size %d exceeds the container bounds (%d bytes)", off, size, d.w64Size)
	}
	if dataSize > math.MaxInt32 {
		return nil, fmt.Errorf("wave64: chunk at offset %d is too large to address (%d bytes of data)", off, dataSize)
	}
	d.w64NextChunkOff = off + int64(padded)

	return &riff.Chunk{
		ID:   wave64ChunkID(guid),
		Size: int(dataSize),
		R:    io.LimitReader(d.r, int64(dataSize)),
	}, nil
}

// readWave64Headers scans the container for the fmt chunk and decodes it.
// It is safe to call multiple times.
func (d *Decoder) readWave64Headers() error {
	if d.NumChans > 0 {
		return nil
	}
	for {
		chunk, err := d.nextWave64Chunk()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return errors.New("wave64: fmt chunk not found")
			}
			return err
		}
		if chunk.ID != riff.FmtID {
			chunk.Drain()
			continue
		}
		if err := d.decodeWave64Fmt(chunk); err != nil {
			return err
		}
		// Rewind the chunk iteration so FwdToPCM and ReadMetadata see
		// every chunk, including the ones preceding fmt.
		d.w64NextChunkOff = wave64HeaderSize
		return nil
	}
}

// decodeWave64Fmt parses the content of a WAVE64 fmt chunk. Only PCM and
// IEEE float content is supported.
func (d *Decoder) decodeWave64Fmt(chunk *riff.Chunk) error {
	if chunk.Size < 16 {
		return fmt.Errorf("wave64: fmt chunk too small (%d bytes, need at least 16)", chunk.Size)
	}
	buf := make([]byte, 16)
	if _, err := io.ReadFull(chunk.R, buf); err != nil {
		return fmt.Errorf("wave64: short read on the fmt chunk: %w", err)
	}
	chunk.Pos += 16

	format := binary.LittleEndian.Uint16(buf[0:2])
	numChans := binary.LittleEndian.Uint16(buf[2:4])
	sampleRate := binary.LittleEndian.Uint32(buf[4:8])
	avgBytesPerSec := binary.LittleEndian.Uint32(buf[8:12])
	blockAlign := binary.LittleEndian.Uint16(buf[12:14])
	bitDepth := binary.LittleEndian.Uint16(buf[14:16])

	if format != wave64FormatPCM && format != wave64FormatIEEEFloat {
		return fmt.Errorf("wave64: unsupported audio format %d, only PCM (%d) and IEEE float (%d) are supported",
			format, wave64FormatPCM, wave64FormatIEEEFloat)
	}
	if numChans < 1 {
		return fmt.Errorf("wave64: invalid channel count %d", numChans)
	}
	if bitDepth < 8 || bitDepth%8 != 0 {
		return fmt.Errorf("wave64: invalid bit depth %d", bitDepth)
	}
	expectedBlockAlign := numChans * (bitDepth / 8)
	if blockAlign != expectedBlockAlign {
		return fmt.Errorf("wave64: invalid block align %d, expected %d for %d channel(s) at %d bits",
			blockAlign, expectedBlockAlign, numChans, bitDepth)
	}

	d.WavAudioFormat = format
	d.NumChans = numChans
	d.SampleRate = sampleRate
	d.AvgBytesPerSec = avgBytesPerSec
	d.BitDepth = bitDepth

	d.parser.WavAudioFormat = format
	d.parser.NumChannels = numChans
	d.parser.SampleRate = sampleRate
	d.parser.AvgBytesPerSec = avgBytesPerSec
	d.parser.BlockAlign = blockAlign
	d.parser.BitsPerSample = bitDepth
	return nil
}
