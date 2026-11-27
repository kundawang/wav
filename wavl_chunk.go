package wav

import (
	"errors"
	"fmt"
	"io"

	"github.com/go-audio/riff"
)

var (
	// CIDWavl is the RIFF form type identifying a Wave List file. Instead of a
	// single "data" chunk, a wavl file stores an ordered sequence of "data"
	// (sound) and "slnt" (silence) chunks that must be merged in order to
	// reconstruct the playable PCM stream.
	CIDWavl = [4]byte{'w', 'a', 'v', 'l'}
	// CIDSlnt is the chunk ID of a silence chunk. Its 4 byte payload is a
	// little-endian uint32 holding the number of silent sample frames
	// (per channel) to synthesize.
	CIDSlnt = [4]byte{'s', 'l', 'n', 't'}

	// ErrWavlNoData is returned when a wavl container does not contain any
	// "data" or "slnt" audio chunk.
	ErrWavlNoData = errors.New("wavl: no data or slnt chunk found")
	// ErrWavlNotPCM is returned when a wavl container uses a non-PCM audio
	// format: the wavl layout is only defined for uncompressed PCM.
	ErrWavlNotPCM = errors.New("wavl: only uncompressed PCM audio is supported")
	// ErrWavlSlntSize is returned when a "slnt" chunk does not carry exactly
	// one uint32 silence count.
	ErrWavlSlntSize = errors.New("wavl: slnt chunk must contain exactly one uint32 sample count")
	// ErrWavlSlntCount is returned when a silence count is invalid, for
	// instance zero or large enough to overflow the decoded PCM size.
	ErrWavlSlntCount = errors.New("wavl: invalid slnt silence sample count")
	// ErrWavlTruncated is returned when an audio chunk promises more bytes
	// than the underlying reader can provide.
	ErrWavlTruncated = errors.New("wavl: truncated audio chunk")
)

// maxWavlPCMBytes caps the decoded PCM size a wavl container may synthesize.
// Silence chunks are only 4 bytes on disk but can claim up to 2^32-1 frames,
// so without a bound a malformed file would make the decoder allocate or emit
// unbounded memory. 1 GiB of decoded PCM is well beyond what the 32 bit slnt
// counter can reach at common frame sizes while still covering any realistic
// in-memory decode.
const maxWavlPCMBytes = uint64(1024 * 1024 * 1024)

// wavlPart describes one entry of the ordered audio chunk sequence of a wavl
// file.
type wavlPart struct {
	id   [4]byte // riff.DataFormatID or CIDSlnt
	size int     // declared chunk size in bytes (padding excluded)
	// padded is the on-disk size including the optional word-alignment byte.
	padded int
	// silence frames carried by a "slnt" chunk (per channel). Only set when
	// id == CIDSlnt.
	silenceFrames uint32
}

// nextWavlChunk reads the next chunk header the same way Decoder.NextChunk
// does, but it also records the declared, unpadded size in
// d.rawChunkSize so the scanner can detect an odd sized "slnt" chunk (whose
// padded size would otherwise be indistinguishable from the required 4 bytes).
func (d *Decoder) nextWavlChunk() (*riff.Chunk, error) {
	var id [4]byte
	var rawSize uint32
	id, rawSize, d.err = d.parser.IDnSize()
	if d.err != nil {
		d.err = fmt.Errorf("error reading chunk header: %w", d.err)
		return nil, d.err
	}
	d.rawChunkSize = int(rawSize)
	padded := rawSize
	if padded%2 == 1 {
		padded++
	}
	return &riff.Chunk{
		ID:   id,
		Size: int(padded),
		R:    io.LimitReader(d.r, int64(padded)),
	}, nil
}

// scanWavl walks the remaining chunks of a wavl container in their on-disk
// order, drains/discards any chunk that is not audio, validates the "slnt"
// chunks and returns the ordered list of audio parts together with the total
// number of PCM bytes the decoded stream will contain. The reader ends up
// positioned right after the last chunk, so a Rewind is required before the
// data can actually be read.
func (d *Decoder) scanWavl() ([]wavlPart, int, error) {
	if d.WavAudioFormat != 0 && d.WavAudioFormat != 1 {
		return nil, 0, fmt.Errorf("%w: format tag %d", ErrWavlNotPCM, d.WavAudioFormat)
	}
	bytesPerFrame := int(d.NumChans) * bytesPerSample(int(d.BitDepth))
	if bytesPerFrame <= 0 {
		return nil, 0, fmt.Errorf("%w: %d channels / %d bits per sample", ErrWavlNotPCM, d.NumChans, d.BitDepth)
	}

	var (
		parts   []wavlPart
		pcmSize uint64
	)
	for {
		chunk, err := d.nextWavlChunk()
		if err != nil {
			// nextWavlChunk wraps the underlying EOF while reading the
			// header; it simply marks the end of the chunk stream.
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, 0, err
		}
		// Declared size without the word-alignment padding byte.
		declared := uint64(d.rawChunkSize)

		switch chunk.ID {
		case riff.DataFormatID:
			if declared%uint64(bytesPerFrame) != 0 {
				return nil, 0, fmt.Errorf("wavl: data chunk of %d bytes is not a multiple of the frame size %d", declared, bytesPerFrame)
			}
			if pcmSize > maxWavlPCMBytes || declared > maxWavlPCMBytes-pcmSize {
				return nil, 0, fmt.Errorf("%w: data chunk overflows the maximum PCM size", ErrWavlSlntCount)
			}
			pcmSize += declared
			parts = append(parts, wavlPart{id: chunk.ID, size: int(declared), padded: chunk.Size})
			chunk.Drain()

		case CIDSlnt:
			// A slnt chunk is exactly one little-endian uint32 count, so its
			// declared size must be 4; an odd declared size is rejected here
			// before word alignment could hide it.
			if d.rawChunkSize != 4 {
				return nil, 0, fmt.Errorf("%w: got %d bytes", ErrWavlSlntSize, d.rawChunkSize)
			}
			var count uint32
			if rerr := chunk.ReadLE(&count); rerr != nil {
				return nil, 0, fmt.Errorf("wavl: failed to read slnt sample count: %w", rerr)
			}
			if count == 0 {
				return nil, 0, fmt.Errorf("%w: zero is not a valid silence length", ErrWavlSlntCount)
			}
			silenceBytes := uint64(count) * uint64(bytesPerFrame)
			if pcmSize > maxWavlPCMBytes || silenceBytes > maxWavlPCMBytes-pcmSize {
				return nil, 0, fmt.Errorf("%w: %d silence frames", ErrWavlSlntCount, count)
			}
			pcmSize += silenceBytes
			parts = append(parts, wavlPart{id: CIDSlnt, size: 4, padded: 4, silenceFrames: count})
			chunk.Drain()

		default:
			// Unknown chunks (LIST, cue, fact, JUNK, ...) are allowed anywhere
			// in the sequence and simply skipped.
			chunk.Drain()
		}
	}

	if len(parts) == 0 {
		return nil, 0, ErrWavlNoData
	}
	return parts, int(pcmSize), nil
}

// wavlStream is an io.Reader materializing the decoded PCM stream of a wavl
// file by walking its "data"/"slnt" chunk sequence in order. "data" chunks are
// streamed from the underlying reader while "slnt" chunks synthesize frames by
// repeating the last frame of sound (or zeroes for a leading silence).
type wavlStream struct {
	d     *Decoder
	parts []wavlPart
	// index of the current audio part and bytes already consumed within it.
	partIndex int
	partPos   int
	// current "data" chunk limited reader, nil while standing on a slnt part.
	cur io.Reader
	// raw bytes of the most recently decoded complete frame, used to hold
	// the sample value across silence chunks. Zero-valued until a full frame
	// has been read.
	lastFrame []byte
	// rolling buffer of the latest data bytes, holding at most the trailing
	// partial frame not yet promoted into lastFrame.
	frameBuf []byte
	// pendingPad is the number of word-alignment bytes still to discard after
	// the current data part's payload has been fully read.
	pendingPad int
	// bytes already written out while expanding the current slnt part.
	silencePos int
}

// newWavlStream prepares a stream over an already scanned wavl file. The
// decoder must be positioned at the start of the chunk stream (right after the
// container/fmt headers), as FwdToPCM leaves it after a Rewind. It opens the
// first audio part, transparently draining any chunk preceding it.
func (d *Decoder) newWavlStream(parts []wavlPart) (*wavlStream, error) {
	frameSize := int(d.NumChans) * bytesPerSample(int(d.BitDepth))
	s := &wavlStream{
		d:         d,
		parts:     parts,
		lastFrame: make([]byte, frameSize),
		frameBuf:  make([]byte, 0, frameSize),
	}
	if err := s.openPart(); err != nil {
		return nil, err
	}
	return s, nil
}

// openPart consumes chunks from the underlying reader until it reaches the
// audio part described by s.parts[s.partIndex], draining any unknown chunk
// found along the way, and prepares its payload reader.
func (s *wavlStream) openPart() error {
	if s.partIndex >= len(s.parts) {
		return io.EOF
	}
	part := s.parts[s.partIndex]

	var chunk *riff.Chunk
	var err error
	for {
		chunk, err = s.d.NextChunk()
		if err != nil {
			return err
		}
		if chunk.ID == part.id {
			break
		}
		// Interspersed metadata/unknown chunk, drain and keep looking.
		chunk.Drain()
	}

	switch part.id {
	case CIDSlnt:
		// scanWavl already validated the shape; just consume the payload so
		// the underlying reader stays positioned on the next chunk header.
		if chunk.Size != 4 {
			return fmt.Errorf("%w: got %d bytes", ErrWavlSlntSize, chunk.Size)
		}
		chunk.Drain()
		s.cur = nil
		s.silencePos = 0
		s.partPos = 0
		return nil
	default:
		// Limit the reader to the declared bytes, excluding any
		// word-alignment padding byte which is consumed after the payload.
		s.cur = io.LimitReader(s.d.r, int64(part.size))
		s.partPos = 0
		s.pendingPad = part.padded - part.size
		return nil
	}
}

// advance moves to the next audio part and opens it, which resets the
// per-part partPos and silencePos counters.
func (s *wavlStream) advance() error {
	s.partIndex++
	s.partPos = 0
	s.cur = nil
	return s.openPart()
}

// Read implements io.Reader. It always fills p when possible, transparently
// chaining the data and synthesized silence chunks and handling reads that
// straddle chunk boundaries or are smaller than a single frame.
func (s *wavlStream) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	written := 0
	for written < len(p) {
		if s.partIndex >= len(s.parts) {
			return written, io.EOF
		}
		part := s.parts[s.partIndex]

		if part.id == CIDSlnt {
			// silencePos starts at 0 when openPart entered this slnt part and
			// grows across Read calls until the whole silence is emitted.
			// Expand the silence by repeating the last frame bytes.
			frameSize := len(s.lastFrame)
			if uint64(part.silenceFrames)*uint64(frameSize) > maxWavlPCMBytes {
				return written, fmt.Errorf("%w: %d silence frames", ErrWavlSlntCount, part.silenceFrames)
			}
			silenceBytes := int(uint64(part.silenceFrames) * uint64(frameSize))
			n := len(p) - written
			if remaining := silenceBytes - s.silencePos; n > remaining {
				n = remaining
			}
			// In-frame position at the start of this write, so a new frame
			// always restarts on its first sample even when previous Read
			// calls ended mid-frame.
			frameOffset := s.silencePos % frameSize
			for i := 0; i < n; i++ {
				if frameOffset == frameSize {
					frameOffset = 0
				}
				p[written+i] = s.lastFrame[frameOffset]
				frameOffset++
			}
			written += n
			s.silencePos += n
			s.partPos += n
			if s.silencePos == silenceBytes {
				if err := s.advance(); err != nil && !errors.Is(err, io.EOF) {
					return written, err
				}
			}
			continue
		}

		// "data" part: stream bytes from the underlying limited reader opened
		// by openPart/advance.
		n, err := s.cur.Read(p[written:])
		if n > 0 {
			s.captureLastFrame(p[written : written+n])
			written += n
			s.partPos += n
		}
		if err == io.EOF {
			if s.partPos != part.size {
				// The limited reader ended early: the file did not contain
				// the full declared chunk.
				return written, fmt.Errorf("%w: expected %d bytes, got %d", ErrWavlTruncated, part.size, s.partPos)
			}
			// Discard the optional word-alignment padding byte so the next
			// NextChunk call reads the following chunk header, not the pad.
			if s.pendingPad > 0 {
				pad := make([]byte, s.pendingPad)
				if _, perr := io.ReadFull(s.d.r, pad); perr != nil {
					return written, fmt.Errorf("%w: missing padding byte: %w", ErrWavlTruncated, perr)
				}
				s.pendingPad = 0
			}
			if aerr := s.advance(); aerr != nil {
				if errors.Is(aerr, io.EOF) {
					// Either the whole container ended (last part) or the
					// next chunk header is missing because the file is cut
					// short while another audio part was declared.
					if s.partIndex+1 < len(s.parts) {
						return written, fmt.Errorf("%w: missing chunk %d of %d", ErrWavlTruncated, s.partIndex+2, len(s.parts))
					}
					return written, io.EOF
				}
				return written, aerr
			}
			continue
		}
		if err != nil {
			return written, err
		}
	}
	return written, nil
}

// captureLastFrame records the most recent complete frame from freshly read
// data bytes. frameBuf retains at most one frame worth of bytes: whenever it
// holds a full frame, those bytes become lastFrame and are dropped so the next
// bytes build the following frame. Partial frames stay buffered and are
// completed by the next read, making the result independent of how reads split
// across frame or chunk boundaries.
func (s *wavlStream) captureLastFrame(b []byte) {
	frameSize := len(s.lastFrame)
	if frameSize == 0 {
		return
	}
	s.frameBuf = append(s.frameBuf, b...)
	for len(s.frameBuf) >= frameSize {
		copy(s.lastFrame, s.frameBuf[:frameSize])
		// Drop the consumed frame; any remainder starts the next one.
		s.frameBuf = s.frameBuf[frameSize:]
	}
}
