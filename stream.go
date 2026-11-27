package wav

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"time"

	"github.com/go-audio/audio"
	"github.com/go-audio/riff"
)

const (
	// A PCM fmt chunk only needs 16 bytes and the WAVE_FORMAT_EXTENSIBLE
	// variant 40 bytes. The size is bounded when reading the chunk in memory
	// so a bogus header can't make the decoder allocate an arbitrary amount
	// of memory.
	maxStreamFmtChunkSize = 1 << 20

	// Format tags understood by the streaming decoder. Compressed formats and
	// IEEE float are rejected with ErrUnsupportedStreamFormat.
	waveFormatPCM        = 0x0001
	waveFormatExtensible = 0xfffe

	// Some writers emitting a stream whose final size isn't known yet (live
	// capture, HTTP preview endpoints, ...) use 0xFFFFFFFF as the data chunk
	// size. Such a payload is read until the underlying reader reaches EOF.
	unknownChunkSize = int64(0xffffffff)
)

var (
	// ErrNotWAVStream is returned when the stream doesn't start with a
	// RIFF/WAVE header.
	ErrNotWAVStream = errors.New("wav: stream is not a RIFF/WAVE container")
	// ErrUnsupportedStreamFormat is returned when the fmt chunk describes a
	// format the streaming decoder can't decode as integer PCM (compressed
	// data, IEEE float, an unexpected number of bits per sample, ...).
	ErrUnsupportedStreamFormat = errors.New("wav: unsupported WAV stream format")
	// ErrInvalidStreamFmt is returned when the fmt chunk is present but
	// contains inconsistent or unusable parameters.
	ErrInvalidStreamFmt = errors.New("wav: invalid fmt chunk in stream")
	// ErrInvalidStreamChunkSize is returned when a chunk header declares a
	// size that can't be valid for its content.
	ErrInvalidStreamChunkSize = errors.New("wav: invalid chunk size in stream")
	// ErrStreamFmtMissing is returned when a data chunk is reached before a
	// usable fmt chunk.
	ErrStreamFmtMissing = errors.New("wav: fmt chunk missing before data chunk")
	// ErrStreamNotSeekable is returned by StreamDecoder.Seek and
	// StreamDecoder.Rewind: a streaming decoder only moves forward.
	ErrStreamNotSeekable = errors.New("wav: streaming decoder is not seekable")
	// ErrStreamTruncated is returned when the underlying reader ends before
	// the declared content was fully read. Errors returned for this condition
	// also match io.ErrUnexpectedEOF with errors.Is.
	ErrStreamTruncated = errors.New("wav: unexpected end of stream")
)

// Compatibility names for the errors above, so callers can use whichever
// spelling matches their code base.
var (
	ErrUnsupportedFormat = ErrUnsupportedStreamFormat
	ErrInvalidFmtChunk   = ErrInvalidStreamFmt
	ErrInvalidChunkSize  = ErrInvalidStreamChunkSize
	ErrTruncatedStream   = ErrStreamTruncated
	ErrNotSeekable       = ErrStreamNotSeekable
)

// truncatedStreamError reports an early end of stream. It deliberately
// matches both ErrStreamTruncated (and its aliases) and io.ErrUnexpectedEOF so
// callers can check either sentinel with errors.Is.
type truncatedStreamError struct {
	detail string
}

func (e *truncatedStreamError) Error() string {
	return fmt.Sprintf("%s: %s: %v", ErrStreamTruncated, e.detail, io.ErrUnexpectedEOF)
}

func (e *truncatedStreamError) Is(target error) bool {
	switch target {
	case ErrStreamTruncated, ErrTruncatedStream, io.ErrUnexpectedEOF:
		return true
	}
	return false
}

func truncatedStream(detail string) error {
	return &truncatedStreamError{detail: detail}
}

// StreamDecoder decodes a RIFF/WAVE stream in a single forward pass from an
// io.Reader. It is meant for sources that can't seek, such as an HTTP body or
// a FIFO: no part of the PCM payload is buffered before it is returned to the
// caller, so playback can start as soon as the headers and the first samples
// have arrived.
//
// Unlike Decoder, a StreamDecoder can't be rewound or seeked; Seek and Rewind
// return ErrStreamNotSeekable. The sample values, channel interleaving and
// bit depth semantics are the same as Decoder.PCMBuffer.
type StreamDecoder struct {
	r io.Reader

	// Container information copied from the fmt chunk.
	NumChans       uint16
	BitDepth       uint16
	SampleRate     uint32
	AvgBytesPerSec uint32
	BlockAlign     uint16
	WavAudioFormat uint16

	// PCMSize is the size in bytes declared by the data chunk header, or -1
	// when the stream uses the 0xFFFFFFFF "unknown size" marker.
	PCMSize int
	// RIFFSize is the size value from the RIFF header (the whole container
	// minus the 8 byte RIFF/ID size prefix).
	RIFFSize uint32

	err error

	riffRead    bool
	fmtRead     bool
	dataFound   bool
	dataEnded   bool
	unknownSize bool
	pcmAccessed bool

	// pcmRemaining is the number of data chunk bytes not read yet (only
	// meaningful when unknownSize is false).
	pcmRemaining int64

	// scratch buffer reused across reads.
	tmp []byte
}

// NewStreamDecoder creates a single pass decoder reading a RIFF/WAVE stream
// from r. It never calls Seek and it doesn't read the PCM payload ahead of the
// caller. Parse errors are reported by the first decoding call or by Err.
func NewStreamDecoder(r io.Reader) *StreamDecoder {
	return &StreamDecoder{r: r}
}

// NewStreamingDecoder is an alias for NewStreamDecoder.
func NewStreamingDecoder(r io.Reader) *StreamDecoder {
	return NewStreamDecoder(r)
}

// StreamingDecoder is an alias for StreamDecoder.
type StreamingDecoder = StreamDecoder

func (d *StreamDecoder) setErr(err error) error {
	if err != nil && d.err == nil {
		d.err = err
	}
	return err
}

// Err returns the first non-EOF error encountered by the decoder.
func (d *StreamDecoder) Err() error {
	if d == nil || errors.Is(d.err, io.EOF) {
		return nil
	}
	return d.err
}

// EOF returns positively if the whole data chunk was consumed.
func (d *StreamDecoder) EOF() bool {
	if d == nil || errors.Is(d.err, io.EOF) {
		return true
	}
	return false
}

// SampleBitDepth returns the bit depth of each sample.
func (d *StreamDecoder) SampleBitDepth() int32 {
	if d == nil {
		return 0
	}
	return int32(d.BitDepth)
}

// PCMLen returns the number of bytes declared by the data chunk header, or -1
// if the stream doesn't declare a size.
func (d *StreamDecoder) PCMLen() int64 {
	if d == nil {
		return 0
	}
	return int64(d.PCMSize)
}

// Format returns the audio format described by the fmt chunk, or nil if the
// headers haven't been parsed yet.
func (d *StreamDecoder) Format() *audio.Format {
	if d == nil || d.NumChans == 0 {
		return nil
	}
	return &audio.Format{
		NumChannels: int(d.NumChans),
		SampleRate:  int(d.SampleRate),
	}
}

// WasPCMAccessed returns positively if the data chunk was located.
func (d *StreamDecoder) WasPCMAccessed() bool {
	if d == nil {
		return false
	}
	return d.pcmAccessed
}

// Seek always fails with ErrStreamNotSeekable: a StreamDecoder only moves
// forward.
func (d *StreamDecoder) Seek(offset int64, whence int) (int64, error) {
	return 0, fmt.Errorf("%w: Seek(%d, %d)", ErrStreamNotSeekable, offset, whence)
}

// Rewind always fails with ErrStreamNotSeekable: the PCM data already returned
// can't be read again. Open a new stream instead.
func (d *StreamDecoder) Rewind() error {
	return fmt.Errorf("%w: Rewind", ErrStreamNotSeekable)
}

// ReadInfo parses the container header and the fmt chunk, leaving the reader
// positioned right after the fmt chunk. It is safe to call multiple times and
// doesn't consume any PCM data.
func (d *StreamDecoder) ReadInfo() {
	if d == nil {
		return
	}
	if err := d.advanceToFmt(); err != nil {
		return
	}
}

// FwdToPCM walks the chunk list until the data chunk is found and leaves the
// reader at the first PCM byte. Unknown chunks are skipped.
func (d *StreamDecoder) FwdToPCM() error {
	if d == nil {
		return ErrPCMChunkNotFound
	}
	return d.advanceToData()
}

// IsValidFile reports whether the stream contains a decodable PCM data chunk.
// The PCM payload itself is not consumed.
func (d *StreamDecoder) IsValidFile() bool {
	if d == nil {
		return false
	}
	if err := d.advanceToData(); err != nil {
		return false
	}
	return d.NumChans > 0 && d.SampleRate > 0 && d.BitDepth >= 8 && d.PCMSize != 0
}

func (d *StreamDecoder) readContainerHeader() error {
	if d.err != nil {
		return d.err
	}
	if d.riffRead {
		return nil
	}
	if d.r == nil {
		return d.setErr(fmt.Errorf("%w: nil reader", ErrNotWAVStream))
	}

	var hdr [12]byte
	if _, err := io.ReadFull(d.r, hdr[:]); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return d.setErr(truncatedStream("reading the RIFF/WAVE header"))
		}
		return d.setErr(err)
	}
	if string(hdr[0:4]) != "RIFF" {
		return d.setErr(fmt.Errorf("%w: got RIFF ID %q", ErrNotWAVStream, hdr[0:4]))
	}
	if string(hdr[8:12]) != "WAVE" {
		return d.setErr(fmt.Errorf("%w: got form type %q", ErrUnsupportedStreamFormat, hdr[8:12]))
	}
	d.RIFFSize = binary.LittleEndian.Uint32(hdr[4:8])
	// The RIFF size covers the rest of the container after the size field.
	// A size of 0 or the unknown marker means the container length wasn't
	// known when the header was written; chunk walking then simply stops at
	// the chunk level (data chunk size, EOF, ...).
	if d.RIFFSize == 0 || int64(d.RIFFSize) == unknownChunkSize {
		d.RIFFSize = 0
	}
	d.riffRead = true
	return nil
}

// nextChunkHeader reads an 8 byte chunk header. It returns io.EOF only when
// the stream ends cleanly at a chunk boundary.
func (d *StreamDecoder) nextChunkHeader() (id [4]byte, size uint32, err error) {
	var hdr [8]byte
	if _, err := io.ReadFull(d.r, hdr[:]); err != nil {
		if errors.Is(err, io.EOF) {
			return id, 0, io.EOF
		}
		if errors.Is(err, io.ErrUnexpectedEOF) {
			return id, 0, truncatedStream("reading a chunk header")
		}
		return id, 0, err
	}
	copy(id[:], hdr[0:4])
	size = binary.LittleEndian.Uint32(hdr[4:8])
	return id, size, nil
}

func (d *StreamDecoder) readPadByte() error {
	var pad [1]byte
	if _, err := io.ReadFull(d.r, pad[:]); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return truncatedStream("reading an odd-sized chunk's padding byte")
		}
		return err
	}
	return nil
}

func (d *StreamDecoder) skipChunk(id [4]byte, size uint32) error {
	if int64(size) == unknownChunkSize {
		return fmt.Errorf("%w: chunk %q declares the unknown size marker", ErrInvalidStreamChunkSize, id)
	}
	if size > 0 {
		if _, err := io.CopyN(ioutil.Discard, d.r, int64(size)); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return truncatedStream(fmt.Sprintf("skipping the %q chunk (%d bytes)", id, size))
			}
			return err
		}
	}
	if size%2 == 1 {
		if err := d.readPadByte(); err != nil {
			return err
		}
	}
	return nil
}

// readFmtChunk reads and validates a fmt chunk of the given payload size.
func (d *StreamDecoder) readFmtChunk(size uint32) error {
	if size < 16 {
		return fmt.Errorf("%w: fmt chunk is %d bytes, need at least 16", ErrInvalidStreamChunkSize, size)
	}
	if size > maxStreamFmtChunkSize {
		return fmt.Errorf("%w: fmt chunk is %d bytes, maximum is %d", ErrInvalidStreamChunkSize, size, maxStreamFmtChunkSize)
	}
	body := make([]byte, int(size))
	if _, err := io.ReadFull(d.r, body); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return truncatedStream(fmt.Sprintf("reading the %d byte fmt chunk", size))
		}
		return err
	}
	if size%2 == 1 {
		if err := d.readPadByte(); err != nil {
			return err
		}
	}

	formatTag := binary.LittleEndian.Uint16(body[0:2])
	numChans := binary.LittleEndian.Uint16(body[2:4])
	sampleRate := binary.LittleEndian.Uint32(body[4:8])
	avgBytesPerSec := binary.LittleEndian.Uint32(body[8:12])
	blockAlign := binary.LittleEndian.Uint16(body[12:14])
	bitDepth := binary.LittleEndian.Uint16(body[14:16])

	switch formatTag {
	case waveFormatPCM:
		// Supported as-is.
	case waveFormatExtensible:
		// The actual format tag is the first 16 bits of the SubFormat GUID.
		if size < 40 {
			return fmt.Errorf("%w: WAVE_FORMAT_EXTENSIBLE fmt chunk is %d bytes, need 40", ErrInvalidStreamFmt, size)
		}
		subFormat := binary.LittleEndian.Uint16(body[24:26])
		if subFormat != waveFormatPCM {
			return fmt.Errorf("%w: WAVE_FORMAT_EXTENSIBLE sub-format 0x%04x", ErrUnsupportedStreamFormat, subFormat)
		}
	default:
		return fmt.Errorf("%w: format tag 0x%04x", ErrUnsupportedStreamFormat, formatTag)
	}

	switch bitDepth {
	case 8, 16, 24, 32:
	default:
		return fmt.Errorf("%w: %d bits per sample", ErrUnsupportedStreamFormat, bitDepth)
	}
	if numChans == 0 {
		return fmt.Errorf("%w: %d channels", ErrInvalidStreamFmt, numChans)
	}
	if sampleRate == 0 {
		return fmt.Errorf("%w: sample rate is 0", ErrInvalidStreamFmt)
	}
	// BlockAlign is recorded for callers but not enforced: like Decoder, the
	// sample width and channel count are what drive decoding, and some
	// writers emit a zero or stale block align without corrupting the PCM.

	d.NumChans = numChans
	d.BitDepth = bitDepth
	d.SampleRate = sampleRate
	d.AvgBytesPerSec = avgBytesPerSec
	d.BlockAlign = blockAlign
	d.WavAudioFormat = formatTag
	d.fmtRead = true
	return nil
}

func (d *StreamDecoder) advanceToFmt() error {
	if d.err != nil {
		return d.err
	}
	if d.fmtRead {
		return nil
	}
	if err := d.readContainerHeader(); err != nil {
		return err
	}
	for {
		id, size, err := d.nextChunkHeader()
		if errors.Is(err, io.EOF) {
			return d.setErr(ErrStreamFmtMissing)
		}
		if err != nil {
			return d.setErr(err)
		}
		switch id {
		case riff.FmtID:
			if err := d.readFmtChunk(size); err != nil {
				return d.setErr(err)
			}
			return nil
		case riff.DataFormatID:
			return d.setErr(ErrStreamFmtMissing)
		default:
			if err := d.skipChunk(id, size); err != nil {
				return d.setErr(err)
			}
		}
	}
}

func (d *StreamDecoder) advanceToData() error {
	if d == nil {
		return ErrPCMChunkNotFound
	}
	if d.dataFound {
		// A clean end-of-stream is reported by PCMBuffer as "no more
		// samples", not as an error.
		if errors.Is(d.err, io.EOF) {
			return nil
		}
		return d.err
	}
	if d.err != nil {
		return d.err
	}
	if err := d.advanceToFmt(); err != nil {
		return err
	}
	for {
		id, size, err := d.nextChunkHeader()
		if errors.Is(err, io.EOF) {
			return d.setErr(ErrPCMChunkNotFound)
		}
		if err != nil {
			return d.setErr(err)
		}
		if id == riff.FmtID {
			// A second fmt chunk is redundant; skip it.
			if err := d.skipChunk(id, size); err != nil {
				return d.setErr(err)
			}
			continue
		}
		if id != riff.DataFormatID {
			if err := d.skipChunk(id, size); err != nil {
				return d.setErr(err)
			}
			continue
		}

		if int64(size) == unknownChunkSize {
			d.unknownSize = true
			d.PCMSize = -1
			d.pcmRemaining = 0
		} else {
			d.PCMSize = int(size)
			d.pcmRemaining = int64(size)
		}
		d.dataFound = true
		d.pcmAccessed = true
		return nil
	}
}

func (d *StreamDecoder) finishData() {
	d.dataEnded = true
	d.pcmRemaining = 0
	// The optional RIFF padding byte following an odd-sized data chunk is
	// deliberately left unread: the streaming decoder stops at the end of the
	// PCM payload, and reading the byte would block a live source that
	// hasn't sent it yet.
	if d.err == nil {
		d.err = io.EOF
	}
}

// decodeSamples converts whole little-endian PCM samples from src into dst and
// returns the number of samples written.
func (d *StreamDecoder) decodeSamples(dst []int, src []byte) int {
	bps := bytesPerSample(int(d.BitDepth))
	if bps <= 0 {
		return 0
	}
	n := len(src) / bps
	if n > len(dst) {
		n = len(dst)
	}
	switch d.BitDepth {
	case 8:
		for i := 0; i < n; i++ {
			dst[i] = int(src[i])
		}
	case 16:
		for i := 0; i < n; i++ {
			dst[i] = int(int16(binary.LittleEndian.Uint16(src[i*2:])))
		}
	case 24:
		for i := 0; i < n; i++ {
			dst[i] = int(audio.Int24LETo32(src[i*3:]))
		}
	case 32:
		for i := 0; i < n; i++ {
			dst[i] = int(int32(binary.LittleEndian.Uint32(src[i*4:])))
		}
	}
	return n
}

// PCMBuffer fills the passed buffer with up to len(buf.Data) samples, decoding
// as each underlying read completes so samples are never lost or reordered
// when the source splits its data anywhere. A partial sample at the very end
// of the data chunk is padding and is dropped, matching Decoder.PCMBuffer.
//
// The buffer is filled unless the data chunk ends first, so the number of
// samples returned is stable for a given buffer size. Size the buffer as a
// multiple of the channel count to keep frames whole.
func (d *StreamDecoder) PCMBuffer(buf *audio.IntBuffer) (n int, err error) {
	if d == nil {
		return 0, ErrPCMChunkNotFound
	}
	if buf == nil {
		return 0, nil
	}
	if err := d.advanceToData(); err != nil {
		return 0, err
	}

	// Fill in the format information even when no sample is returned so
	// callers always know what they are decoding.
	buf.Format = d.Format()
	buf.SourceBitDepth = int(d.BitDepth)

	if d.dataEnded || len(buf.Data) == 0 {
		return 0, nil
	}

	bps := bytesPerSample(int(d.BitDepth))
	if bps <= 0 {
		return 0, d.setErr(fmt.Errorf("%w: %d bits per sample", ErrUnsupportedStreamFormat, d.BitDepth))
	}

	for n < len(buf.Data) {
		if !d.unknownSize && d.pcmRemaining <= 0 {
			// All declared payload was consumed.
			d.finishData()
			return n, nil
		}

		want := (len(buf.Data) - n) * bps
		if !d.unknownSize && int64(want) > d.pcmRemaining {
			want = int(d.pcmRemaining)
		}
		if want <= 0 {
			d.finishData()
			return n, nil
		}

		if cap(d.tmp) < want {
			d.tmp = make([]byte, want)
		} else {
			d.tmp = d.tmp[:want]
		}

		// io.ReadFull handles sources that split anywhere, including in the
		// middle of a sample: retrying until the requested bytes arrived
		// keeps sample alignment intact.
		m, readErr := io.ReadFull(d.r, d.tmp[:want])
		if !d.unknownSize {
			d.pcmRemaining -= int64(m)
		}
		if m > 0 {
			n += d.decodeSamples(buf.Data[n:], d.tmp[:m])
		}

		if readErr != nil {
			if errors.Is(readErr, io.EOF) || errors.Is(readErr, io.ErrUnexpectedEOF) {
				if d.unknownSize {
					if m%bps != 0 {
						return n, d.setErr(truncatedStream("data chunk ended in the middle of a sample"))
					}
					d.finishData()
					return n, nil
				}
				// The header promised more PCM data than the reader delivered.
				return n, d.setErr(truncatedStream(fmt.Sprintf("data chunk ended with %d of %d bytes missing", d.pcmRemaining, d.PCMSize)))
			}
			return n, d.setErr(readErr)
		}
	}
	return n, nil
}

// FullPCMBuffer reads the whole PCM payload into memory. Streaming sources
// should prefer PCMBuffer so playback isn't delayed until the end of the
// stream.
func (d *StreamDecoder) FullPCMBuffer() (*audio.IntBuffer, error) {
	if d == nil {
		return nil, ErrPCMChunkNotFound
	}
	if err := d.advanceToData(); err != nil {
		return nil, err
	}
	format := d.Format()
	buf := &audio.IntBuffer{
		Format:         format,
		SourceBitDepth: int(d.BitDepth),
	}
	chunk := make([]int, 4096)
	for {
		scratch := &audio.IntBuffer{
			Data:           chunk,
			Format:         format,
			SourceBitDepth: int(d.BitDepth),
		}
		n, err := d.PCMBuffer(scratch)
		if n > 0 {
			buf.Data = append(buf.Data, scratch.Data[:n]...)
		}
		if err != nil {
			return buf, err
		}
		if n == 0 {
			break
		}
	}
	return buf, nil
}

// Duration returns the duration declared by the container. Like
// Decoder.Duration it is computed from the RIFF container size when that value
// is usable, and from the data chunk size otherwise.
func (d *StreamDecoder) Duration() (time.Duration, error) {
	if d == nil {
		return 0, errors.New("wav: can't calculate the duration of a nil stream decoder")
	}
	if err := d.advanceToData(); err != nil {
		return 0, err
	}
	avg := d.AvgBytesPerSec
	if avg == 0 {
		avg = d.SampleRate * uint32(d.NumChans) * uint32(bytesPerSample(int(d.BitDepth)))
	}
	if avg == 0 {
		return 0, errors.New("wav: can't calculate the duration, missing the average bytes per second")
	}
	size := int64(d.RIFFSize)
	if d.unknownSize || d.PCMSize < 0 {
		// The payload length isn't known, so neither is the duration.
		return 0, errors.New("wav: duration unknown for a stream with an unbounded data chunk")
	}
	if size == 0 {
		size = int64(d.PCMSize)
	}
	return time.Duration(float64(size) / float64(avg) * float64(time.Second)), nil
}

// String implements the Stringer interface.
func (d *StreamDecoder) String() string {
	if d == nil {
		return "wav: nil stream decoder"
	}
	out := "Format: WAVE - "
	if d.NumChans == 0 {
		return out + "headers not read"
	}
	out += fmt.Sprintf("%d channels @ %d / %d bits", d.NumChans, d.SampleRate, d.BitDepth)
	return out
}
