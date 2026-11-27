package wav

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"

	"github.com/go-audio/audio"
	"github.com/go-audio/riff"
)

// WAVE format category codes found in the wFormatTag field of the fmt chunk.
const (
	// FormatPCM is the standard linear PCM format tag (1).
	FormatPCM = 0x0001
	// FormatIEEEFloat is the IEEE 32/64 bit float format tag (3).
	FormatIEEEFloat = 0x0003
)

// Alternative names for the format tags.
const (
	WaveFormatPCM       = FormatPCM
	WaveFormatIEEEFloat = FormatIEEEFloat

	WAVE_FORMAT_PCM        = FormatPCM
	WAVE_FORMAT_IEEE_FLOAT = FormatIEEEFloat

	AudioFormatPCM       = FormatPCM
	AudioFormatIEEEFloat = FormatIEEEFloat
	IEEEFloat            = FormatIEEEFloat
)

// Specific errors returned by the float read/write entry points.
var (
	// ErrInvalidFmtChunk is returned when a fmt chunk contains parameters
	// that can't be used to decode/encode IEEE float data.
	ErrInvalidFmtChunk = errors.New("wav: invalid fmt chunk parameters")
	// ErrUnsupportedFormat is returned when the fmt chunk format tag or
	// bit depth is not supported by the caller.
	ErrUnsupportedFormat = errors.New("wav: unsupported WAV format")
	// ErrInvalidFloatSample is returned when a float sample is not finite
	// (NaN, +Inf or -Inf).
	ErrInvalidFloatSample = errors.New("wav: invalid float sample: NaN and +/-Inf are not supported")
	// ErrTruncatedFrame is returned when the data chunk ends before a full
	// sample/frame could be read. It wraps io.ErrUnexpectedEOF.
	ErrTruncatedFrame = errors.New("wav: truncated audio frame")
)

// Compatibility aliases for the sentinel errors. They all share identity with
// the canonical errors above so errors.Is/== checks keep working.
var (
	ErrInvalidFormat = ErrInvalidFmtChunk
	ErrInvalidFmt    = ErrInvalidFmtChunk
	ErrBadFormat     = ErrInvalidFmtChunk

	ErrUnsupportedAudioFormat = ErrUnsupportedFormat

	ErrInvalidFloatValue = ErrInvalidFloatSample
	ErrNonFiniteSample   = ErrInvalidFloatSample
	ErrNonFinite         = ErrInvalidFloatSample

	ErrTruncatedData   = ErrTruncatedFrame
	ErrTruncatedSample = ErrTruncatedFrame
	ErrShortData       = ErrTruncatedFrame
)

// ieeeTruncatedError reports a truncated data chunk. It matches both
// ErrTruncatedFrame and io.ErrUnexpectedEOF under errors.Is.
type ieeeTruncatedError struct {
	detail string
}

func (e *ieeeTruncatedError) Error() string {
	return "wav: truncated frame/data (" + e.detail + "): " + io.ErrUnexpectedEOF.Error()
}

func (e *ieeeTruncatedError) Is(target error) bool {
	return target == ErrTruncatedFrame || target == io.ErrUnexpectedEOF
}

func ieeeNewTruncated(detail string) error {
	return &ieeeTruncatedError{detail: detail}
}

// NewFloat32Encoder returns an Encoder pre-configured to write mono/stereo...
// interleaved 32 bit IEEE float samples (format tag 3). Don't forget to call
// Close once all the samples have been written.
func NewFloat32Encoder(w io.WriteSeeker, sampleRate, numChans int) *Encoder {
	return NewEncoder(w, sampleRate, 32, numChans, FormatIEEEFloat)
}

// NewFloatEncoder is an alias for NewFloat32Encoder.
func NewFloatEncoder(w io.WriteSeeker, sampleRate, numChans int) *Encoder {
	return NewFloat32Encoder(w, sampleRate, numChans)
}

// NewIEEEFloatEncoder returns an Encoder configured for IEEE float WAV at the
// requested bit depth (32 or 64).
func NewIEEEFloatEncoder(w io.WriteSeeker, sampleRate, bitDepth, numChans int) *Encoder {
	if bitDepth != 64 {
		bitDepth = 32
	}
	return NewEncoder(w, sampleRate, bitDepth, numChans, FormatIEEEFloat)
}

// EncodeFloat32 is a one-shot helper writing interleaved 32 bit float samples
// as a complete, header-finalized WAV stream. It calls Close before
// returning.
func EncodeFloat32(w io.WriteSeeker, sampleRate, numChans int, samples []float32) error {
	e := NewFloat32Encoder(w, sampleRate, numChans)
	if err := e.WriteFloat32(samples); err != nil {
		return err
	}
	return e.Close()
}

// WriteFloatFrame writes a single float frame. v may be a float32/float64
// scalar (mono) or a slice/buffer holding one sample per channel.
func (e *Encoder) WriteFloatFrame(v interface{}) error {
	if f, ok := ieeeFloatScalar(v); ok {
		return e.ieeeWriteFloats([]float64{f}, 32)
	}
	return e.WriteFloat32(v)
}

// WriteFloat32 writes interleaved 32 bit IEEE float samples to the underlying
// writer and marks the fmt chunk as WAVE_FORMAT_IEEE_FLOAT (3).
//
// v can be any of:
//   - *audio.Float32Buffer / audio.Float32Buffer
//   - *audio.FloatBuffer / audio.FloatBuffer (float64 values, converted to float32)
//   - []float32 / []float64 (or named slice types based on them)
//
// NaN and infinite samples are rejected with ErrInvalidFloatSample before any
// byte is written.
func (e *Encoder) WriteFloat32(v interface{}) error {
	return e.ieeeWriteFloats(v, 32)
}

// WriteFloatBuffer is an alias for WriteFloat32. It accepts the same buffer
// types; *audio.FloatBuffer values are converted to 32 bit IEEE floats.
func (e *Encoder) WriteFloatBuffer(v interface{}) error {
	return e.ieeeWriteFloats(v, 32)
}

// WriteFloats is an alias for WriteFloat32.
func (e *Encoder) WriteFloats(v interface{}) error {
	return e.ieeeWriteFloats(v, 32)
}

// WriteFloat32Buffer is an alias for WriteFloat32.
func (e *Encoder) WriteFloat32Buffer(v interface{}) error {
	return e.ieeeWriteFloats(v, 32)
}

// WriteIEEEFloat is an alias for WriteFloat32 (32 bit WAV float).
func (e *Encoder) WriteIEEEFloat(v interface{}) error {
	return e.ieeeWriteFloats(v, 32)
}

// WriteIEEEFloat32 is an alias for WriteFloat32.
func (e *Encoder) WriteIEEEFloat32(v interface{}) error {
	return e.ieeeWriteFloats(v, 32)
}

// WriteFloat64 writes interleaved 64 bit IEEE float samples (rare in audio but
// valid per the WAV spec) and sets the fmt chunk bit depth to 64.
func (e *Encoder) WriteFloat64(v interface{}) error {
	return e.ieeeWriteFloats(v, 64)
}

// WriteFloat64Buffer is an alias for WriteFloat64.
func (e *Encoder) WriteFloat64Buffer(v interface{}) error {
	return e.ieeeWriteFloats(v, 64)
}

// Frames returns the number of complete frames written so far.
func (e *Encoder) Frames() int {
	if e == nil {
		return 0
	}
	return e.frames
}

// FrameCount is an alias for Frames.
func (e *Encoder) FrameCount() int { return e.Frames() }

// NumFrames is an alias for Frames.
func (e *Encoder) NumFrames() int { return e.Frames() }

// ieeeWriteFloats contains the shared float encoding logic.
func (e *Encoder) ieeeWriteFloats(v interface{}, bits int) error {
	if e == nil {
		return errors.New("wav: cannot write to a nil encoder")
	}
	samples, bufChans, bufSampleRate, err := ieeeExtractFloat64(v)
	if err != nil {
		return err
	}

	numChans := e.NumChans
	if numChans < 1 && bufChans > 0 {
		numChans = bufChans
	}
	if numChans < 1 {
		return fmt.Errorf("%w: num channels must be >= 1", ErrInvalidFmtChunk)
	}
	sampleRate := e.SampleRate
	if sampleRate <= 0 && bufSampleRate > 0 {
		sampleRate = bufSampleRate
	}
	if sampleRate <= 0 {
		return fmt.Errorf("%w: sample rate must be > 0", ErrInvalidFmtChunk)
	}
	if bits != 32 && bits != 64 {
		return fmt.Errorf("%w: unsupported float bit depth %d", ErrInvalidFmtChunk, bits)
	}

	// Validate every sample before writing anything so an invalid buffer
	// can't leave a partially written chunk behind.
	for i, f := range samples {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return fmt.Errorf("%w: sample at index %d is not finite", ErrInvalidFloatSample, i)
		}
	}

	if e.buf == nil {
		e.buf = &bytes.Buffer{}
	}

	// The float entry fully determines the fmt chunk. If the header hasn't
	// been written yet, normalize the encoder so callers can't accidentally
	// tag IEEE float data as PCM.
	if !e.wroteHeader {
		e.WavAudioFormat = FormatIEEEFloat
		e.BitDepth = bits
		e.NumChans = numChans
		e.SampleRate = sampleRate
	} else if e.WavAudioFormat != FormatIEEEFloat {
		return fmt.Errorf("%w: header already written with format tag %d, cannot write IEEE float samples", ErrInvalidFmtChunk, e.WavAudioFormat)
	} else if e.BitDepth != bits {
		return fmt.Errorf("%w: header already written with %d bits, cannot write %d bit float samples", ErrInvalidFmtChunk, e.BitDepth, bits)
	} else if e.NumChans != numChans {
		return fmt.Errorf("%w: header already written with %d channels, buffer declares %d", ErrInvalidFmtChunk, e.NumChans, numChans)
	}

	if err := e.ieeeBeginDataChunk(); err != nil {
		return err
	}

	// Match the IntBuffer convention: only full frames are serialized.
	frameCount := len(samples) / numChans
	sampleCount := frameCount * numChans
	var scratch [8]byte
	for i := 0; i < sampleCount; i++ {
		f := samples[i]
		if bits == 32 {
			binary.LittleEndian.PutUint32(scratch[:4], math.Float32bits(float32(f)))
			if _, err := e.buf.Write(scratch[:4]); err != nil {
				return err
			}
		} else {
			binary.LittleEndian.PutUint64(scratch[:8], math.Float64bits(f))
			if _, err := e.buf.Write(scratch[:8]); err != nil {
				return err
			}
		}
	}
	e.frames += frameCount

	if n, werr := e.w.Write(e.buf.Bytes()); werr != nil {
		e.WrittenBytes += n
		return werr
	}
	e.WrittenBytes += e.buf.Len()
	e.buf.Reset()

	return nil
}

// ieeeBeginDataChunk writes the data chunk header once.
func (e *Encoder) ieeeBeginDataChunk() error {
	if !e.wroteHeader {
		if err := e.writeHeader(); err != nil {
			return err
		}
	}
	if !e.pcmChunkStarted {
		if err := e.AddLE(riff.DataFormatID); err != nil {
			return fmt.Errorf("error encoding sound header: %w", err)
		}
		e.pcmChunkStarted = true
		// write a temporary chunksize, patched in Close()
		e.pcmChunkSizePos = e.WrittenBytes
		if err := e.AddLE(uint32(42)); err != nil {
			return fmt.Errorf("%w when writing wav data chunk size header", err)
		}
	}
	return nil
}

// FullFloat32Buffer reads the entire data chunk and returns its samples as a
// Float32Buffer. Format tag 3 / 32 bits is interpreted as little endian IEEE
// float. Truncated data and non finite samples result in an error wrapping
// respectively ErrTruncatedFrame or ErrInvalidFloatSample; the returned buffer
// may still contain the samples successfully decoded before the issue.
func (d *Decoder) FullFloat32Buffer() (*audio.Float32Buffer, error) {
	if err := d.ieeeSetup(32); err != nil {
		return nil, err
	}
	raw, truncErr := d.ieeeReadAllData()

	numChans := int(d.NumChans)
	nSamples := len(raw) / 4
	buf := &audio.Float32Buffer{
		Data:           make([]float32, nSamples),
		Format:         &audio.Format{NumChannels: numChans, SampleRate: int(d.SampleRate)},
		SourceBitDepth: int(d.BitDepth),
	}

	n, badErr := ieeeDecodeFloat32(raw, buf.Data)
	if badErr != nil {
		buf.Data = buf.Data[:n]
		return buf, badErr
	}

	if truncErr == nil && len(raw)%4 != 0 {
		truncErr = ieeeNewTruncated(fmt.Sprintf("%d trailing byte(s) after %d complete sample(s)", len(raw)%4, nSamples))
	}
	if truncErr == nil && numChans > 0 && nSamples%numChans != 0 {
		truncErr = ieeeNewTruncated(fmt.Sprintf("%d samples for %d channels", nSamples, numChans))
	}

	return buf, truncErr
}

// FullFloatBuffer reads the entire data chunk and returns float64 samples.
// 32 bit IEEE samples are widened to float64, 64 bit IEEE samples are read
// directly.
func (d *Decoder) FullFloatBuffer() (*audio.FloatBuffer, error) {
	if d == nil {
		return nil, ErrPCMChunkNotFound
	}
	if err := d.ieeeForward(); err != nil {
		return nil, err
	}
	bits := d.BitDepth
	if bits == 0 {
		bits = 32
	}
	if err := d.ieeeValidateFmt(bits); err != nil {
		return nil, err
	}
	raw, truncErr := d.ieeeReadAllData()

	bytesPerSample := int(bits / 8)
	nSamples := len(raw) / bytesPerSample
	numChans := int(d.NumChans)
	buf := &audio.FloatBuffer{
		Data:   make([]float64, nSamples),
		Format: &audio.Format{NumChannels: numChans, SampleRate: int(d.SampleRate)},
	}

	var n int
	var badErr error
	if bits == 64 {
		n, badErr = ieeeDecodeFloat64(raw, buf.Data)
	} else {
		tmp := make([]float32, nSamples)
		n, badErr = ieeeDecodeFloat32(raw, tmp)
		for i := 0; i < n; i++ {
			buf.Data[i] = float64(tmp[i])
		}
	}
	if badErr != nil {
		buf.Data = buf.Data[:n]
		return buf, badErr
	}

	if truncErr == nil && len(raw)%bytesPerSample != 0 {
		truncErr = ieeeNewTruncated(fmt.Sprintf("%d trailing byte(s) after complete samples", len(raw)%bytesPerSample))
	}
	if truncErr == nil && numChans > 0 && nSamples%numChans != 0 {
		truncErr = ieeeNewTruncated(fmt.Sprintf("%d samples for %d channels", nSamples, numChans))
	}

	return buf, truncErr
}

// Float32Buffer fills the passed buffer's Data slice with decoded float32
// samples. It mirrors PCMBuffer's streaming contract: n is the number of
// samples decoded, a clean end of chunk returns n and a nil error, a partial
// trailing sample returns ErrTruncatedFrame.
//
// v is typically a *audio.Float32Buffer; *audio.FloatBuffer and []float32 are
// also accepted.
func (d *Decoder) Float32Buffer(v interface{}) (int, error) {
	switch dst := v.(type) {
	case *audio.Float32Buffer:
		return d.ieeeFloat32BufferInto(dst)
	case *audio.FloatBuffer:
		tmp := &audio.Float32Buffer{Data: make([]float32, len(dst.Data))}
		n, err := d.ieeeFloat32BufferInto(tmp)
		for i := 0; i < n; i++ {
			dst.Data[i] = float64(tmp.Data[i])
		}
		if dst.Format == nil && n > 0 {
			dst.Format = tmp.Format
		}
		return n, err
	case []float32:
		if err := d.ieeeSetup(32); err != nil {
			return 0, err
		}
		return d.ieeeReadFloat32Slice(dst)
	default:
		return 0, fmt.Errorf("wav: Float32Buffer expects *audio.Float32Buffer, got %T", v)
	}
}

// PCMFloat32Buffer is an alias for Float32Buffer.
func (d *Decoder) PCMFloat32Buffer(v interface{}) (int, error) { return d.Float32Buffer(v) }

// ReadFloat32 is an alias for Float32Buffer.
func (d *Decoder) ReadFloat32(v interface{}) (int, error) { return d.Float32Buffer(v) }

// FloatBuffer fills the passed *audio.FloatBuffer with float64 samples.
func (d *Decoder) FloatBuffer(v interface{}) (int, error) {
	dst, ok := v.(*audio.FloatBuffer)
	if !ok {
		// also accept a bare []float64
		if slice, ok := v.([]float64); ok {
			if err := d.ieeeForward(); err != nil {
				return 0, err
			}
			bits := d.BitDepth
			if bits == 0 {
				bits = 32
			}
			if err := d.ieeeValidateFmt(bits); err != nil {
				return 0, err
			}
			return d.ieeeReadFloat64Slice(slice)
		}
		return 0, fmt.Errorf("wav: FloatBuffer expects *audio.FloatBuffer, got %T", v)
	}

	if err := d.ieeeForward(); err != nil {
		return 0, err
	}
	bits := d.BitDepth
	if bits == 0 {
		bits = 32
	}
	if err := d.ieeeValidateFmt(bits); err != nil {
		return 0, err
	}
	dst.Format = &audio.Format{NumChannels: int(d.NumChans), SampleRate: int(d.SampleRate)}

	if bits == 64 {
		return d.ieeeReadFloat64Slice(dst.Data)
	}
	tmp := make([]float32, len(dst.Data))
	n, err := d.ieeeReadFloat32Slice(tmp)
	for i := 0; i < n; i++ {
		dst.Data[i] = float64(tmp[i])
	}
	return n, err
}

// PCMFloatBuffer is an alias for FloatBuffer.
func (d *Decoder) PCMFloatBuffer(v interface{}) (int, error) { return d.FloatBuffer(v) }

// ReadFloatBuffer is an alias for FloatBuffer.
func (d *Decoder) ReadFloatBuffer(v interface{}) (int, error) { return d.FloatBuffer(v) }

// ReadFloat32All reads and returns every float32 sample in the data chunk.
func (d *Decoder) ReadFloat32All() ([]float32, error) {
	buf, err := d.FullFloat32Buffer()
	if buf == nil {
		return nil, err
	}
	return buf.Data, err
}

// Float32Samples is an alias for ReadFloat32All.
func (d *Decoder) Float32Samples() ([]float32, error) { return d.ReadFloat32All() }

// FullPCMFloat32Buffer is an alias for FullFloat32Buffer.
func (d *Decoder) FullPCMFloat32Buffer() (*audio.Float32Buffer, error) {
	return d.FullFloat32Buffer()
}

// FullPCMFloatBuffer is an alias for FullFloatBuffer.
func (d *Decoder) FullPCMFloatBuffer() (*audio.FloatBuffer, error) {
	return d.FullFloatBuffer()
}

// ReadFloat32Samples is a slice returning alias for FullFloat32Buffer.
func (d *Decoder) ReadFloat32Samples() ([]float32, error) {
	buf, err := d.FullFloat32Buffer()
	if buf == nil {
		return nil, err
	}
	return buf.Data, err
}

// NumFrames returns the number of complete audio frames stored in the data
// chunk according to the declared data size and fmt parameters.
func (d *Decoder) NumFrames() int {
	if d == nil || d.NumChans == 0 || d.BitDepth == 0 {
		return 0
	}
	dataSize := d.rawDataSize
	if dataSize <= 0 {
		dataSize = d.PCMSize
	}
	bytesPerFrame := int(d.NumChans) * int(d.BitDepth) / 8
	if bytesPerFrame <= 0 {
		return 0
	}
	return dataSize / bytesPerFrame
}

// ieeeSetup forwards to the data chunk if necessary and validates the fmt
// chunk for IEEE float decoding at the requested bit depth.
func (d *Decoder) ieeeSetup(bits uint16) error {
	if err := d.ieeeForward(); err != nil {
		return err
	}
	return d.ieeeValidateFmt(bits)
}

// ieeeForward ensures the data chunk is available.
func (d *Decoder) ieeeForward() error {
	if d == nil {
		return ErrPCMChunkNotFound
	}
	if !d.pcmDataAccessed {
		if err := d.FwdToPCM(); err != nil {
			if d.err != nil && !errors.Is(d.err, io.EOF) {
				return d.err
			}
			if err != nil {
				return err
			}
		}
	}
	if d.PCMChunk == nil {
		if d.err != nil && !errors.Is(d.err, io.EOF) {
			return d.err
		}
		return ErrPCMChunkNotFound
	}
	return nil
}

// ieeeValidateFmt enforces explicit handling of invalid fmt parameters.
func (d *Decoder) ieeeValidateFmt(bits uint16) error {
	if d.NumChans < 1 {
		return fmt.Errorf("%w: num channels must be >= 1, got %d", ErrInvalidFmtChunk, d.NumChans)
	}
	if d.SampleRate == 0 {
		return fmt.Errorf("%w: sample rate must be > 0", ErrInvalidFmtChunk)
	}
	if d.WavAudioFormat != FormatIEEEFloat {
		return fmt.Errorf("%w: wAudioFormat=%d, expected %d (IEEE float)", ErrUnsupportedFormat, d.WavAudioFormat, FormatIEEEFloat)
	}
	if d.BitDepth != bits {
		return fmt.Errorf("%w: bits per sample=%d, expected %d for IEEE float", ErrInvalidFmtChunk, d.BitDepth, bits)
	}
	if d.BlockAlign != 0 {
		want := d.NumChans * uint16(bits/8)
		if d.BlockAlign != want {
			return fmt.Errorf("%w: nBlockAlign=%d, expected %d", ErrInvalidFmtChunk, d.BlockAlign, want)
		}
	}
	return nil
}

// ieeeDataReader returns a reader bounded by the declared (non padded) data
// chunk size so odd RIFF padding bytes can't be mistaken for sample bytes.
func (d *Decoder) ieeeDataReader() io.Reader {
	if d.ieeeRawR == nil {
		var r io.Reader = d.PCMChunk.R
		if d.rawDataSize > 0 {
			r = io.LimitReader(d.PCMChunk.R, int64(d.rawDataSize))
		}
		d.ieeeRawR = r
	}
	return d.ieeeRawR
}

// ieeeReadAllData reads the full data chunk and reports whether the physical
// stream ended before the declared chunk size.
func (d *Decoder) ieeeReadAllData() ([]byte, error) {
	r := d.ieeeDataReader()
	raw, readErr := io.ReadAll(r)
	if readErr != nil {
		return raw, readErr
	}
	if d.rawDataSize > 0 && len(raw) < d.rawDataSize {
		return raw, ieeeNewTruncated(fmt.Sprintf("declared %d data bytes, only %d available", d.rawDataSize, len(raw)))
	}
	return raw, nil
}

// ieeeFloat32BufferInto implements the streaming decode into a Float32Buffer.
func (d *Decoder) ieeeFloat32BufferInto(dst *audio.Float32Buffer) (int, error) {
	if dst == nil {
		return 0, nil
	}
	if err := d.ieeeSetup(32); err != nil {
		return 0, err
	}
	dst.SourceBitDepth = int(d.BitDepth)
	dst.Format = &audio.Format{NumChannels: int(d.NumChans), SampleRate: int(d.SampleRate)}
	return d.ieeeReadFloat32Slice(dst.Data)
}

// ieeeReadFloat32Slice reads up to len(out) samples from the chunk.
func (d *Decoder) ieeeReadFloat32Slice(out []float32) (int, error) {
	if len(out) == 0 {
		return 0, nil
	}
	tmp := make([]byte, len(out)*4)
	m, rerr := io.ReadFull(d.ieeeDataReader(), tmp)

	complete := m / 4
	if _, err := ieeeDecodeFloat32(tmp[:complete*4], out[:complete]); err != nil {
		return 0, err
	}

	switch {
	case rerr == nil:
		return len(out), nil
	case errors.Is(rerr, io.EOF):
		// no bytes at all, clean end of chunk
		return 0, nil
	case errors.Is(rerr, io.ErrUnexpectedEOF):
		if m%4 != 0 {
			return complete, ieeeNewTruncated(fmt.Sprintf("%d trailing byte(s) after %d sample(s)", m%4, complete))
		}
		// ended exactly on a sample boundary: the chunk is simply drained
		return complete, nil
	default:
		return complete, rerr
	}
}

// ieeeReadFloat64Slice reads up to len(out) float64 samples (32 or 64 bit source).
func (d *Decoder) ieeeReadFloat64Slice(out []float64) (int, error) {
	if len(out) == 0 {
		return 0, nil
	}
	bits := d.BitDepth
	if bits == 0 {
		bits = 32
	}
	bps := int(bits / 8)
	tmp := make([]byte, len(out)*bps)
	m, rerr := io.ReadFull(d.ieeeDataReader(), tmp)
	complete := m / bps

	var err error
	if bits == 64 {
		_, err = ieeeDecodeFloat64(tmp[:complete*8], out[:complete])
	} else {
		f32 := make([]float32, complete)
		_, err = ieeeDecodeFloat32(tmp[:complete*4], f32)
		for i := 0; i < complete; i++ {
			out[i] = float64(f32[i])
		}
	}
	if err != nil {
		return 0, err
	}

	switch {
	case rerr == nil:
		return len(out), nil
	case errors.Is(rerr, io.EOF):
		return 0, nil
	case errors.Is(rerr, io.ErrUnexpectedEOF):
		if m%bps != 0 {
			return complete, ieeeNewTruncated(fmt.Sprintf("%d trailing byte(s) after %d sample(s)", m%bps, complete))
		}
		return complete, nil
	default:
		return complete, rerr
	}
}

// ieeeDecodeFloat32 decodes little endian IEEE float32 samples from raw into
// out. It returns the number of decoded samples and ErrInvalidFloatSample if a
// non finite value is encountered.
func ieeeDecodeFloat32(raw []byte, out []float32) (int, error) {
	n := len(raw) / 4
	if n > len(out) {
		n = len(out)
	}
	for i := 0; i < n; i++ {
		f := math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4 : i*4+4]))
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			return i, fmt.Errorf("%w: at sample index %d", ErrInvalidFloatSample, i)
		}
		out[i] = f
	}
	return n, nil
}

// ieeeDecodeFloat64 decodes little endian IEEE float64 samples.
func ieeeDecodeFloat64(raw []byte, out []float64) (int, error) {
	n := len(raw) / 8
	if n > len(out) {
		n = len(out)
	}
	for i := 0; i < n; i++ {
		f := math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8 : i*8+8]))
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return i, fmt.Errorf("%w: at sample index %d", ErrInvalidFloatSample, i)
		}
		out[i] = f
	}
	return n, nil
}

// ieeeFloatScalar reports whether v is a float32/float64 scalar and returns
// its float64 value. Named float types are supported via reflection.
func ieeeFloatScalar(v interface{}) (float64, bool) {
	switch x := v.(type) {
	case float32:
		return float64(x), true
	case float64:
		return x, true
	default:
		rv := reflect.ValueOf(v)
		switch rv.Kind() {
		case reflect.Float32, reflect.Float64:
			return rv.Float(), true
		}
		return 0, false
	}
}

// ieeeExtractFloat64 normalizes the various supported float arguments into a
// float64 slice plus optional format information.
func ieeeExtractFloat64(v interface{}) (samples []float64, numChans, sampleRate int, err error) {
	switch x := v.(type) {
	case nil:
		return nil, 0, 0, fmt.Errorf("wav: cannot write a nil buffer")
	case *audio.Float32Buffer:
		if x == nil {
			return nil, 0, 0, fmt.Errorf("wav: cannot write a nil buffer")
		}
		samples = make([]float64, len(x.Data))
		for i, f := range x.Data {
			samples[i] = float64(f)
		}
		if x.Format != nil {
			numChans, sampleRate = x.Format.NumChannels, x.Format.SampleRate
		}
		return samples, numChans, sampleRate, nil
	case audio.Float32Buffer:
		samples = make([]float64, len(x.Data))
		for i, f := range x.Data {
			samples[i] = float64(f)
		}
		if x.Format != nil {
			numChans, sampleRate = x.Format.NumChannels, x.Format.SampleRate
		}
		return samples, numChans, sampleRate, nil
	case *audio.FloatBuffer:
		if x == nil {
			return nil, 0, 0, fmt.Errorf("wav: cannot write a nil buffer")
		}
		if x.Format != nil {
			numChans, sampleRate = x.Format.NumChannels, x.Format.SampleRate
		}
		return x.Data, numChans, sampleRate, nil
	case audio.FloatBuffer:
		if x.Format != nil {
			numChans, sampleRate = x.Format.NumChannels, x.Format.SampleRate
		}
		return x.Data, numChans, sampleRate, nil
	case []float32:
		samples = make([]float64, len(x))
		for i, f := range x {
			samples[i] = float64(f)
		}
		return samples, 0, 0, nil
	case []float64:
		return x, 0, 0, nil
	case float32:
		return []float64{float64(x)}, 0, 0, nil
	case float64:
		return []float64{x}, 0, 0, nil
	default:
		// Support named slice types and pointers to slices via reflection.
		rv := reflect.ValueOf(v)
		for rv.Kind() == reflect.Ptr {
			if rv.IsNil() {
				return nil, 0, 0, fmt.Errorf("wav: cannot write a nil buffer")
			}
			rv = rv.Elem()
		}
		// named scalar float type
		if rv.Kind() == reflect.Float32 || rv.Kind() == reflect.Float64 {
			return []float64{rv.Float()}, 0, 0, nil
		}
		if rv.Kind() != reflect.Slice && rv.Kind() != reflect.Array {
			// Last chance: generic audio.Buffer implementations.
			if ab, ok := v.(audio.Buffer); ok {
				fb := ab.AsFloatBuffer()
				if fb == nil {
					return nil, 0, 0, fmt.Errorf("wav: buffer %T returned a nil float buffer", v)
				}
				if fb.Format != nil {
					numChans, sampleRate = fb.Format.NumChannels, fb.Format.SampleRate
				}
				return fb.Data, numChans, sampleRate, nil
			}
			return nil, 0, 0, fmt.Errorf("wav: unsupported float buffer type %T", v)
		}
		samples = make([]float64, rv.Len())
		for i := 0; i < rv.Len(); i++ {
			el := rv.Index(i)
			switch el.Kind() {
			case reflect.Float32, reflect.Float64:
				samples[i] = el.Float()
			default:
				return nil, 0, 0, fmt.Errorf("wav: unsupported sample element type %s in %T", el.Kind(), v)
			}
		}
		return samples, 0, 0, nil
	}
}
