package wav

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"time"

	"github.com/go-audio/audio"
	"github.com/go-audio/riff"
)

// Encoder encodes LPCM data into a wav containter.
type Encoder struct {
	w   io.WriteSeeker
	buf *bytes.Buffer

	SampleRate int
	BitDepth   int
	NumChans   int

	// A number indicating the WAVE format category of the file. The content of
	// the <format-specific-fields> portion of the ‘fmt’ chunk, and the
	// interpretation of the waveform data, depend on this value. PCM = 1 (i.e.
	// Linear quantization) Values other than 1 indicate some form of
	// compression.
	WavAudioFormat int

	// Metadata contains metadata to inject in the file.
	Metadata *Metadata

	WrittenBytes    int
	frames          int
	pcmChunkStarted bool
	pcmChunkSizePos int
	wroteHeader     bool // true if we've written the header out

	// wave64 selects the Sony Wave64 container (GUIDs + 64 bit chunk sizes)
	// instead of the classic 32 bit RIFF/WAVE layout. It is set by
	// NewWave64Encoder.
	wave64 bool
	// dataSizePos records the offset of the Wave64 data chunk size field so
	// it can be backfilled on Close.
	dataSizePos int64
	// closed is set by Close so a second call is a no-op instead of
	// backfilling the size fields a second time.
	closed bool
}

// bytesBufferFor preallocates the staging buffer used while serializing PCM
// samples, roughly one minute of audio.
func bytesBufferFor(sampleRate, bitDepth, numChans int) *bytes.Buffer {
	return bytes.NewBuffer(make([]byte, 0, bytesNumFromDuration(time.Minute, sampleRate, bitDepth)*numChans))
}

// NewEncoder creates a new encoder to create a new wav file.
// Don't forget to add Frames to the encoder before writing.
func NewEncoder(w io.WriteSeeker, sampleRate, bitDepth, numChans, audioFormat int) *Encoder {
	return &Encoder{
		w:              w,
		buf:            bytesBufferFor(sampleRate, bitDepth, numChans),
		SampleRate:     sampleRate,
		BitDepth:       bitDepth,
		NumChans:       numChans,
		WavAudioFormat: audioFormat,
	}
}

// AddLE serializes and adds the passed value using little endian
func (e *Encoder) AddLE(src interface{}) error {
	e.WrittenBytes += binary.Size(src)
	return binary.Write(e.w, binary.LittleEndian, src)
}

// AddBE serializes and adds the passed value using big endian
func (e *Encoder) AddBE(src interface{}) error {
	e.WrittenBytes += binary.Size(src)
	return binary.Write(e.w, binary.BigEndian, src)
}

func (e *Encoder) addBuffer(buf *audio.IntBuffer) error {
	if buf == nil {
		return fmt.Errorf("can't add a nil buffer")
	}

	frameCount := buf.NumFrames()
	// performance tweak: setup a buffer so we don't do too many writes
	var err error
	for i := 0; i < frameCount; i++ {
		for j := 0; j < buf.Format.NumChannels; j++ {
			v := buf.Data[i*buf.Format.NumChannels+j]
			switch e.BitDepth {
			case 8:
				if err = binary.Write(e.buf, binary.LittleEndian, uint8(v)); err != nil {
					return err
				}
			case 16:
				if err = binary.Write(e.buf, binary.LittleEndian, int16(v)); err != nil {
					return err
				}
			case 24:
				if err = binary.Write(e.buf, binary.LittleEndian, audio.Int32toInt24LEBytes(int32(v))); err != nil {
					return err
				}
			case 32:
				if err = binary.Write(e.buf, binary.LittleEndian, int32(v)); err != nil {
					return err
				}
			default:
				return fmt.Errorf("can't add frames of bit size %d", e.BitDepth)
			}
		}
		e.frames++
	}
	if n, err := e.w.Write(e.buf.Bytes()); err != nil {
		e.WrittenBytes += n
		return err
	}
	e.WrittenBytes += e.buf.Len()
	e.buf.Reset()

	return nil
}

func (e *Encoder) writeHeader() error {
	if e.wroteHeader {
		return errors.New("already wrote header")
	}
	e.wroteHeader = true
	if e == nil {
		return fmt.Errorf("can't write a nil encoder")
	}
	if e.w == nil {
		return fmt.Errorf("can't write to a nil writer")
	}

	if e.WrittenBytes > 0 {
		return nil
	}

	if e.wave64 {
		return e.writeWave64Header()
	}

	// riff ID
	if err := e.AddLE(riff.RiffID); err != nil {
		return err
	}
	// file size uint32, to update later on.
	if err := e.AddLE(uint32(42)); err != nil {
		return err
	}
	// wave headers
	if err := e.AddLE(riff.WavFormatID); err != nil {
		return err
	}
	// form
	if err := e.AddLE(riff.FmtID); err != nil {
		return err
	}
	// chunk size
	if err := e.AddLE(uint32(16)); err != nil {
		return err
	}
	// wave format
	if err := e.AddLE(uint16(e.WavAudioFormat)); err != nil {
		return err
	}
	// num channels
	if err := e.AddLE(uint16(e.NumChans)); err != nil {
		return fmt.Errorf("error encoding the number of channels - %w", err)
	}
	// samplerate
	if err := e.AddLE(uint32(e.SampleRate)); err != nil {
		return fmt.Errorf("error encoding the sample rate - %w", err)
	}
	blockAlign := e.NumChans * e.BitDepth / 8
	// avg bytes per sec
	if err := e.AddLE(uint32(e.SampleRate * blockAlign)); err != nil {
		return fmt.Errorf("error encoding the avg bytes per sec - %w", err)
	}
	// block align
	if err := e.AddLE(uint16(blockAlign)); err != nil {
		return err
	}
	// bits per sample
	if err := e.AddLE(uint16(e.BitDepth)); err != nil {
		return fmt.Errorf("error encoding bits per sample - %w", err)
	}

	return nil
}

// Write encodes and writes the passed buffer to the underlying writer.
// Don't forget to Close() the encoder or the file won't be valid.
func (e *Encoder) Write(buf *audio.IntBuffer) error {
	if buf == nil {
		return fmt.Errorf("can't add a nil buffer")
	}
	if !e.wroteHeader {
		if err := e.writeHeader(); err != nil {
			return err
		}
	}

	if !e.pcmChunkStarted {
		if e.wave64 {
			if err := e.startWave64PCMChunk(); err != nil {
				return err
			}
			return e.addBuffer(buf)
		}
		// sound header
		if err := e.AddLE(riff.DataFormatID); err != nil {
			return fmt.Errorf("error encoding sound header %w", err)
		}
		e.pcmChunkStarted = true

		// write a temporary chunksize
		e.pcmChunkSizePos = e.WrittenBytes
		if err := e.AddLE(uint32(42)); err != nil {
			return fmt.Errorf("%w when writing wav data chunk size header", err)
		}
	}

	return e.addBuffer(buf)
}

// WriteFrame writes a single frame of data to the underlying writer.
func (e *Encoder) WriteFrame(value interface{}) error {
	if !e.wroteHeader {
		if err := e.writeHeader(); err != nil {
			return err
		}
	}
	if !e.pcmChunkStarted {
		if e.wave64 {
			if err := e.startWave64PCMChunk(); err != nil {
				return err
			}
		} else {
			// sound header
			if err := e.AddLE(riff.DataFormatID); err != nil {
				return fmt.Errorf("error encoding sound header %w", err)
			}
			e.pcmChunkStarted = true

			// write a temporary chunksize
			e.pcmChunkSizePos = e.WrittenBytes
			if err := e.AddLE(uint32(42)); err != nil {
				return fmt.Errorf("%w when writing wav data chunk size header", err)
			}
		}
	}

	e.frames++
	return e.AddLE(value)
}

func (e *Encoder) writeMetadata() error {
	chunkData := encodeInfoChunk(e)
	if err := e.AddBE(CIDList); err != nil {
		return fmt.Errorf("failed to write the LIST chunk ID: %w", err)
	}
	if err := e.AddLE(uint32(len(chunkData))); err != nil {
		return fmt.Errorf("failed to write the LIST chunk size: %w", err)
	}
	return e.AddBE(chunkData)
}

// Close flushes the content to disk, make sure the headers are up to date
// Note that the underlying writer is NOT being closed.
func (e *Encoder) Close() error {
	if e == nil || e.w == nil {
		return nil
	}
	if e.closed {
		return nil
	}
	e.closed = true

	if e.wave64 {
		return e.closeWave64()
	}

	// inject metadata at the end to not trip implementation not supporting
	// metadata chunks
	if e.Metadata != nil {
		if err := e.writeMetadata(); err != nil {
			return fmt.Errorf("failed to write metadata - %w", err)
		}
	}

	// go back and write total size in header
	if _, err := e.w.Seek(4, 0); err != nil {
		return err
	}
	if err := e.AddLE(uint32(e.WrittenBytes) - 8); err != nil {
		return fmt.Errorf("%w when writing the total written bytes", err)
	}

	// rewrite the audio chunk length header
	if e.pcmChunkSizePos > 0 {
		if _, err := e.w.Seek(int64(e.pcmChunkSizePos), 0); err != nil {
			return err
		}
		chunksize := uint64((int(e.BitDepth) / 8) * int(e.NumChans) * e.frames)
		if chunksize > math.MaxUint32 {
			return fmt.Errorf("PCM data of %d bytes overflows the RIFF 32 bit chunk size, use a Wave64 encoder: %w", chunksize, ErrWave64Overflow)
		}
		if err := e.AddLE(uint32(chunksize)); err != nil {
			return fmt.Errorf("%w when writing wav data chunk size header", err)
		}
	}

	// jump back to the end of the file.
	if _, err := e.w.Seek(0, 2); err != nil {
		return err
	}
	switch e.w.(type) {
	case *os.File:
		return e.w.(*os.File).Sync()
	}
	return nil
}

// closeWave64 backfills the 64 bit RIFF and data chunk sizes, emitting an
// empty (but valid) data chunk when no audio frame was ever written.
func (e *Encoder) closeWave64() error {
	if e.Metadata != nil {
		return ErrWave64Metadata
	}

	if !e.wroteHeader {
		if err := e.writeHeader(); err != nil {
			return err
		}
	}
	if !e.pcmChunkStarted {
		// empty audio: still emit a data chunk so the file stays valid.
		if err := e.startWave64PCMChunk(); err != nil {
			return err
		}
	}

	dataChunkSize, err := wave64DataSize(e.frames, e.NumChans, e.BitDepth)
	if err != nil {
		return err
	}

	payloadSize := uint64(e.BitDepth/8) * uint64(e.NumChans) * uint64(e.frames)
	if padding := wave64Padding(payloadSize); padding > 0 {
		// PCM chunks must end on an 8 byte boundary; pad with zero bytes.
		zero := [wave64Alignment]byte{}
		if err := e.addRaw(zero[:padding]); err != nil {
			return err
		}
	}

	// Capture the RIFF size before backfilling anything: the patch helpers do
	// not count overwritten bytes in WrittenBytes, and the size fields must
	// reflect the real file length.
	// In Wave64 the chunk size includes the chunk header itself (GUID + size
	// field), so the RIFF size equals the total file length.
	riffSize := uint64(e.WrittenBytes)
	if riffSize < dataChunkSize {
		return ErrWave64Overflow
	}

	// Backfill the data chunk size (includes its own 24 byte header).
	if _, err := e.w.Seek(e.dataSizePos, io.SeekStart); err != nil {
		return err
	}
	if err := e.patchLE64(dataChunkSize); err != nil {
		return fmt.Errorf("%w when writing Wave64 data chunk size header", err)
	}

	// Backfill the RIFF size.
	if _, err := e.w.Seek(int64(wave64GUIDSize), io.SeekStart); err != nil {
		return err
	}
	if err := e.patchLE64(riffSize); err != nil {
		return fmt.Errorf("%w when writing the Wave64 total size", err)
	}

	// jump back to the end of the file.
	if _, err := e.w.Seek(0, io.SeekEnd); err != nil {
		return err
	}
	switch f := e.w.(type) {
	case *os.File:
		return f.Sync()
	}
	return nil
}
