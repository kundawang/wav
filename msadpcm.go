package wav

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"

	"github.com/go-audio/audio"
	"github.com/go-audio/riff"
)

// WavAudioFormatMSADPCM is the WAVE format code for Microsoft ADPCM.
const WavAudioFormatMSADPCM = 2

// CIDFact is the chunk ID for the "fact" chunk, which stores the number of
// sample frames contained in the data chunk for compressed formats such as
// Microsoft ADPCM.
var CIDFact = [4]byte{'f', 'a', 'c', 't'}

const (
	// msADPCMHeaderSize is the size in bytes of the per channel block header:
	// predictor (1 byte), delta (2 bytes), sample1 (2 bytes), sample2 (2 bytes).
	// A stereo block therefore has a 14 byte header.
	msADPCMHeaderSize = 7

	msADPCMMaxSample = 1<<15 - 1
	msADPCMMinSample = -1 << 15
	// msADPCMMaxDelta mirrors the overflow guard used by other MS ADPCM
	// decoders so that a malformed stream cannot overflow the step size.
	msADPCMMaxDelta = (1<<31 - 1) / 768
)

// msADPCMAdaptationTable is the 16 entry adaptation table from the Microsoft
// ADPCM specification. It is indexed by the unsigned 4 bit differential code.
var msADPCMAdaptationTable = [16]int{
	230, 230, 230, 230, 307, 409, 512, 614,
	768, 614, 512, 409, 307, 230, 230, 230,
}

// adpcmCoef is a pair of signed prediction coefficients.
type adpcmCoef struct {
	c1, c2 int
}

// msADPCMFmt is the format information carried by the fmt chunk extension for
// Microsoft ADPCM files.
type msADPCMFmt struct {
	blockAlign      int
	samplesPerBlock int
	coefs           []adpcmCoef
}

// msADPCMChanState is the per channel decoder state while expanding a block.
type msADPCMChanState struct {
	sample1 int
	sample2 int
	delta   int
}

// expand decodes a single differential code and returns the reconstructed
// sample. Note that the predictor division truncates towards zero, matching
// the C reference decoders (and Go's integer division semantics).
func (cs *msADPCMChanState) expand(coef adpcmCoef, nibble int) int {
	predictor := (cs.sample1*coef.c1 + cs.sample2*coef.c2) / 256
	predictor += nibble * cs.delta
	if predictor > msADPCMMaxSample {
		predictor = msADPCMMaxSample
	} else if predictor < msADPCMMinSample {
		predictor = msADPCMMinSample
	}
	cs.sample2 = cs.sample1
	cs.sample1 = predictor

	// The adaptation table is indexed by the unsigned 4 bit code, so mask
	// the signed nibble to recover it.
	cs.delta = (msADPCMAdaptationTable[nibble&0x0f] * cs.delta) >> 8
	if cs.delta < 16 {
		cs.delta = 16
	}
	if cs.delta > msADPCMMaxDelta {
		cs.delta = msADPCMMaxDelta
	}
	return predictor
}

// msADPCMState holds the streaming state used to decode the data chunk.
type msADPCMState struct {
	fmt      *msADPCMFmt
	channels int

	// factFrames is the total number of frames announced by the fact chunk.
	// It is only used when factSet is true and factFrames is greater than 0.
	factFrames int64
	factSet    bool

	// remaining is the number of bytes left in the data chunk.
	remaining int
	// leftover holds bytes received during a read that ended before a full
	// block could be assembled, so they are consumed by the next call.
	leftover []byte

	// pending contains the decoded interleaved samples for the current block
	// that haven't been handed to the caller yet.
	pending []int
	pos     int

	framesOut int64
	done      bool
	err       error
}

// decodeFmtChunk parses the fmt chunk and, for Microsoft ADPCM, the format
// extension. It replaces riff.Chunk.DecodeWavHeader in Decoder.readHeaders so
// that the extension (block size, samples per block and prediction
// coefficients) is available to the decoder.
//
// For formats other than Microsoft ADPCM the behavior is unchanged: the first
// 16 bytes populate the parser and any remaining bytes are discarded.
func (d *Decoder) decodeFmtChunk(chunk *riff.Chunk) error {
	if chunk == nil {
		return errors.New("can't decode a nil fmt chunk")
	}
	if chunk.Size < 16 {
		return fmt.Errorf("fmt chunk too short: %d bytes, need at least 16", chunk.Size)
	}
	var base [16]byte
	if _, err := io.ReadFull(chunk, base[:]); err != nil {
		return fmt.Errorf("failed to read the fmt chunk: %w", err)
	}

	d.parser.WavAudioFormat = binary.LittleEndian.Uint16(base[0:2])
	d.parser.NumChannels = binary.LittleEndian.Uint16(base[2:4])
	d.parser.SampleRate = binary.LittleEndian.Uint32(base[4:8])
	d.parser.AvgBytesPerSec = binary.LittleEndian.Uint32(base[8:12])
	d.parser.BlockAlign = binary.LittleEndian.Uint16(base[12:14])
	d.parser.BitsPerSample = binary.LittleEndian.Uint16(base[14:16])

	d.NumChans = d.parser.NumChannels
	d.BitDepth = d.parser.BitsPerSample
	d.SampleRate = d.parser.SampleRate
	d.WavAudioFormat = d.parser.WavAudioFormat
	d.AvgBytesPerSec = d.parser.AvgBytesPerSec

	remaining := chunk.Size - 16
	if d.WavAudioFormat != WavAudioFormatMSADPCM {
		// Preserve the previous behavior for PCM and unsupported formats:
		// consume the optional extension without interpreting it.
		chunk.Drain()
		return nil
	}

	if remaining < 2 {
		return errors.New("MS ADPCM fmt chunk is missing its extension")
	}
	var cbBuf [2]byte
	if _, err := io.ReadFull(chunk, cbBuf[:]); err != nil {
		return fmt.Errorf("failed to read the MS ADPCM fmt extension size: %w", err)
	}
	cbSize := int(binary.LittleEndian.Uint16(cbBuf[:]))
	if cbSize < 4 {
		return fmt.Errorf("invalid MS ADPCM fmt extension size: %d", cbSize)
	}
	if cbSize > remaining-2 {
		return fmt.Errorf("MS ADPCM fmt extension size %d exceeds the %d bytes available", cbSize, remaining-2)
	}

	ext := make([]byte, cbSize)
	if _, err := io.ReadFull(chunk, ext); err != nil {
		return fmt.Errorf("failed to read the MS ADPCM fmt extension: %w", err)
	}
	// Consume any bytes left in the chunk (including a possible padding byte)
	// so the reader stays aligned on the next chunk.
	chunk.Drain()

	samplesPerBlock := int(binary.LittleEndian.Uint16(ext[0:2]))
	numCoef := int(binary.LittleEndian.Uint16(ext[2:4]))
	if numCoef < 1 {
		return fmt.Errorf("invalid MS ADPCM coefficient count: %d", numCoef)
	}
	if 4+4*numCoef > cbSize {
		return fmt.Errorf("MS ADPCM fmt extension is too small for %d coefficients", numCoef)
	}
	coefs := make([]adpcmCoef, numCoef)
	for i := range coefs {
		off := 4 + 4*i
		coefs[i] = adpcmCoef{
			c1: int(int16(binary.LittleEndian.Uint16(ext[off : off+2]))),
			c2: int(int16(binary.LittleEndian.Uint16(ext[off+2 : off+4]))),
		}
	}

	if d.BitDepth != 4 {
		return fmt.Errorf("MS ADPCM must declare 4 bits per sample, got %d", d.BitDepth)
	}
	if d.NumChans != 1 && d.NumChans != 2 {
		return fmt.Errorf("MS ADPCM is only supported for mono and stereo streams, got %d channels", d.NumChans)
	}
	blockAlign := int(d.parser.BlockAlign)
	channels := int(d.NumChans)
	if blockAlign < msADPCMHeaderSize*channels {
		return fmt.Errorf("MS ADPCM block alignment %d is too small for %d channel(s)", blockAlign, channels)
	}
	// A block can never hold more samples than its size allows. The declared
	// samples per block is authoritative (it is what the encoder used), but it
	// can't exceed the room the block alignment leaves for differential codes.
	if samplesPerBlock < 2 {
		return fmt.Errorf("invalid MS ADPCM samples per block: %d", samplesPerBlock)
	}
	if capacity := msADPCMSamplesPerBlock(blockAlign, channels); samplesPerBlock > capacity {
		return fmt.Errorf("MS ADPCM samples per block %d exceeds the %d samples a %d byte block can hold",
			samplesPerBlock, capacity, blockAlign)
	}

	d.msFmt = &msADPCMFmt{
		blockAlign:      blockAlign,
		samplesPerBlock: samplesPerBlock,
		coefs:           coefs,
	}
	return nil
}

// msADPCMSamplesPerBlock returns the number of sample frames contained in a
// full Microsoft ADPCM block.
func msADPCMSamplesPerBlock(blockAlign, channels int) int {
	if channels < 1 {
		return 0
	}
	return 2 + (blockAlign-msADPCMHeaderSize*channels)*2/channels
}

// decodeFactChunk reads the total number of sample frames from a fact chunk.
func (d *Decoder) decodeFactChunk(chunk *riff.Chunk) error {
	if chunk == nil {
		return errors.New("can't decode a nil fact chunk")
	}
	if chunk.Size < 4 {
		return fmt.Errorf("fact chunk too short: %d bytes, need at least 4", chunk.Size)
	}
	var b [4]byte
	if _, err := io.ReadFull(chunk, b[:]); err != nil {
		return fmt.Errorf("failed to read the fact chunk: %w", err)
	}
	// Discard any trailing bytes (some writers pad the chunk).
	chunk.Drain()
	d.msFactFrames = int64(binary.LittleEndian.Uint32(b[:]))
	d.msFactSet = true
	return nil
}

// initMSADPCM prepares the streaming decoder state. It is safe to call
// multiple times.
func (d *Decoder) initMSADPCM() error {
	if d.msState != nil {
		return nil
	}
	if d.msFactErr != nil {
		return d.msFactErr
	}
	if d.msFmt == nil {
		return errors.New("MS ADPCM format information is missing from the fmt chunk")
	}
	if d.PCMChunk == nil {
		return ErrPCMChunkNotFound
	}
	dataSize := d.pcmDataSize
	if dataSize <= 0 {
		dataSize = d.PCMSize
	}
	d.msState = &msADPCMState{
		fmt:        d.msFmt,
		channels:   int(d.NumChans),
		factFrames: d.msFactFrames,
		factSet:    d.msFactSet,
		remaining:  dataSize,
	}
	return nil
}

// pcmBufferMSADPCM fills buf with decoded Microsoft ADPCM samples.
// buf.SourceBitDepth is set to 16 because compressed samples are expanded to
// 16 bit signed PCM.
func (d *Decoder) pcmBufferMSADPCM(buf *audio.IntBuffer) (n int, err error) {
	if err := d.initMSADPCM(); err != nil {
		return 0, err
	}
	m := d.msState
	buf.Format = d.Format()
	buf.SourceBitDepth = 16

	n = 0
	for n < len(buf.Data) {
		if m.pos < len(m.pending) {
			copied := copy(buf.Data[n:], m.pending[m.pos:])
			n += copied
			m.pos += copied
			continue
		}
		m.pending = nil
		m.pos = 0
		if m.err != nil {
			return n, m.err
		}
		if m.done {
			break
		}
		if err := m.decodeNextBlock(d.PCMChunk.R); err != nil {
			m.err = err
			return n, err
		}
	}
	return n, m.err
}

// fullPCMBufferMSADPCM decodes the entire data chunk.
func (d *Decoder) fullPCMBufferMSADPCM() (*audio.IntBuffer, error) {
	if err := d.initMSADPCM(); err != nil {
		return nil, err
	}
	out := &audio.IntBuffer{
		Data:           make([]int, 0, 4096),
		Format:         d.Format(),
		SourceBitDepth: 16,
	}
	tmp := make([]int, 4096)
	for {
		buf := &audio.IntBuffer{Data: tmp, Format: d.Format()}
		n, err := d.pcmBufferMSADPCM(buf)
		if n > 0 {
			out.Data = append(out.Data, tmp[:n]...)
		}
		if err != nil {
			return nil, err
		}
		if n == 0 {
			break
		}
	}
	return out, nil
}

// decodeNextBlock reads and expands the next block of the data chunk.
func (m *msADPCMState) decodeNextBlock(r io.Reader) error {
	if m.done {
		return nil
	}
	if m.factSet && m.factFrames > 0 && m.framesOut >= m.factFrames {
		m.done = true
		return nil
	}
	if m.remaining <= 0 {
		if m.factSet && m.factFrames > 0 && m.framesOut < m.factFrames {
			return fmt.Errorf("MS ADPCM data is truncated: fact declares %d frames but only %d are available",
				m.factFrames, m.framesOut)
		}
		m.done = true
		return nil
	}

	// Assemble the block from any bytes carried over from the previous call
	// plus what the reader can provide.
	blockLen := m.fmt.blockAlign
	if blockLen > m.remaining {
		// The last block of a stream is allowed to be shorter than the
		// regular block alignment.
		blockLen = m.remaining
	}
	block := make([]byte, blockLen)
	copied := copy(block, m.leftover)
	m.leftover = nil
	n, err := io.ReadFull(r, block[copied:])
	if err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return fmt.Errorf("MS ADPCM block is truncated: got %d of %d bytes: %w",
				copied+n, blockLen, err)
		}
		return fmt.Errorf("failed to read an MS ADPCM block: %w", err)
	}
	m.remaining -= blockLen

	frames, err := m.decodeBlock(block)
	if err != nil {
		return err
	}
	// A full block holds exactly the declared number of sample frames. Any
	// trailing differential codes are ignored, matching the header.
	if maxFrames := m.fmt.samplesPerBlock; len(frames)/m.channels > maxFrames {
		frames = frames[:maxFrames*m.channels]
	}

	// The fact chunk gives the exact number of sample frames. The last block
	// of a stream is frequently only partially filled, so only keep the
	// frames announced by the fact chunk.
	if m.factSet && m.factFrames > 0 {
		left := m.factFrames - m.framesOut
		if left < 0 {
			left = 0
		}
		if int64(len(frames)/m.channels) > left {
			frames = frames[:int(left)*m.channels]
		}
	}

	m.pending = frames
	m.pos = 0
	m.framesOut += int64(len(frames) / m.channels)

	if m.remaining <= 0 && m.factSet && m.factFrames > 0 && m.framesOut < m.factFrames {
		return fmt.Errorf("MS ADPCM data is truncated: fact declares %d frames but only %d are available",
			m.factFrames, m.framesOut)
	}
	return nil
}

// decodeBlock expands a single (possibly short) block into interleaved samples.
func (m *msADPCMState) decodeBlock(block []byte) ([]int, error) {
	channels := m.channels
	headerSize := msADPCMHeaderSize * channels
	if len(block) < headerSize {
		return nil, fmt.Errorf("MS ADPCM block is truncated: %d bytes, need at least %d header bytes",
			len(block), headerSize)
	}

	coefs := m.fmt.coefs
	state := make([]msADPCMChanState, channels)

	// Per channel block header: predictor, delta, sample1, sample2. The
	// values for all channels are stored channel by channel.
	predictors := make([]int, channels)
	for c := 0; c < channels; c++ {
		predictors[c] = int(block[c])
		if predictors[c] >= len(coefs) {
			return nil, fmt.Errorf("MS ADPCM block has predictor %d out of range [0,%d)",
				predictors[c], len(coefs))
		}
	}
	off := channels
	for c := 0; c < channels; c++ {
		delta := int(int16(binary.LittleEndian.Uint16(block[off : off+2])))
		if delta <= 0 {
			return nil, fmt.Errorf("MS ADPCM block has invalid delta %d", delta)
		}
		state[c].delta = delta
		off += 2
	}
	for c := 0; c < channels; c++ {
		state[c].sample1 = int(int16(binary.LittleEndian.Uint16(block[off : off+2])))
		off += 2
	}
	for c := 0; c < channels; c++ {
		state[c].sample2 = int(int16(binary.LittleEndian.Uint16(block[off : off+2])))
		off += 2
	}

	// The two header samples are emitted sample2 first, then sample1 for
	// every channel (frame by frame, not channel by channel).
	out := make([]int, 0, 2*channels+len(block)-headerSize)
	for c := 0; c < channels; c++ {
		out = append(out, state[c].sample2)
	}
	for c := 0; c < channels; c++ {
		out = append(out, state[c].sample1)
	}

	if channels == 1 {
		for _, b := range block[headerSize:] {
			out = append(out, state[0].expand(coefs[predictors[0]], signedNibble(b>>4)))
			out = append(out, state[0].expand(coefs[predictors[0]], signedNibble(b&0x0f)))
		}
	} else {
		// Stereo samples are interleaved: the high nibble belongs to the
		// left channel, the low nibble to the right channel.
		for _, b := range block[headerSize:] {
			out = append(out, state[0].expand(coefs[predictors[0]], signedNibble(b>>4)))
			out = append(out, state[1].expand(coefs[predictors[1]], signedNibble(b&0x0f)))
		}
	}
	return out, nil
}

// signedNibble returns the signed value of a 4 bit differential code, in the
// range [-8, 7].
func signedNibble(b byte) int {
	if b&0x08 != 0 {
		return int(b) - 16
	}
	return int(b)
}
