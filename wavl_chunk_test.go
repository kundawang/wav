package wav

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"reflect"
	"testing"

	"github.com/go-audio/audio"
	"github.com/go-audio/riff"
)

// wavlBuilder assures RIFF/wavl containers in memory. It keeps full control of
// the on-disk layout so tests can craft truncated chunks, odd padding, unknown
// chunks and interleaved data/slnt sequences.
type wavlBuilder struct {
	formType   [4]byte
	numChans   uint16
	bitDepth   uint16
	sampleRate uint32
	formatTag  uint16
	body       bytes.Buffer
}

func newWavlBuilder(numChans, bitDepth int, sampleRate int) *wavlBuilder {
	return &wavlBuilder{
		formType:   CIDWavl,
		numChans:   uint16(numChans),
		bitDepth:   uint16(bitDepth),
		sampleRate: uint32(sampleRate),
		formatTag:  1, // WAVE_FORMAT_PCM
	}
}

func (b *wavlBuilder) chunk(id [4]byte, payload []byte) {
	b.body.Write(id[:])
	binary.Write(&b.body, binary.LittleEndian, uint32(len(payload)))
	b.body.Write(payload)
	if len(payload)%2 == 1 {
		b.body.WriteByte(0) // word alignment padding
	}
}

// dataChunk writes a sound chunk from interleaved int samples (one value per
// channel sample), serialized at the container bit depth.
func (b *wavlBuilder) dataChunk(samples []int) {
	payload := encodeSamplesLE(samples, int(b.bitDepth))
	b.chunk(riff.DataFormatID, payload)
}

// slntChunk writes a silence chunk carrying the requested frame count.
func (b *wavlBuilder) slntChunk(frames uint32) {
	payload := make([]byte, 4)
	binary.LittleEndian.PutUint32(payload, frames)
	b.chunk(CIDSlnt, payload)
}

// rawSlntChunk writes a slnt chunk with an arbitrary payload, used to exercise
// malformed chunk sizes.
func (b *wavlBuilder) rawSlntChunk(payload []byte) {
	b.chunk(CIDSlnt, payload)
}

func (b *wavlBuilder) unknownChunk(id [4]byte, payload []byte) {
	b.chunk(id, payload)
}

func (b *wavlBuilder) bytes() []byte {
	var out bytes.Buffer
	out.Write(riff.RiffID[:])
	// size of everything after the RIFF header (form type + chunks)
	binary.Write(&out, binary.LittleEndian, uint32(4+b.body.Len()))
	out.Write(b.formType[:])

	blockAlign := b.numChans * b.bitDepth / 8
	avgBytesPerSec := b.sampleRate * uint32(blockAlign)
	var fmtBuf bytes.Buffer
	binary.Write(&fmtBuf, binary.LittleEndian, b.formatTag)
	binary.Write(&fmtBuf, binary.LittleEndian, b.numChans)
	binary.Write(&fmtBuf, binary.LittleEndian, b.sampleRate)
	binary.Write(&fmtBuf, binary.LittleEndian, avgBytesPerSec)
	binary.Write(&fmtBuf, binary.LittleEndian, blockAlign)
	binary.Write(&fmtBuf, binary.LittleEndian, b.bitDepth)
	out.Write(riff.FmtID[:])
	binary.Write(&out, binary.LittleEndian, uint32(fmtBuf.Len()))
	out.Write(fmtBuf.Bytes())

	out.Write(b.body.Bytes())
	return out.Bytes()
}

// truncate drops the last n bytes of the container.
func (b *wavlBuilder) truncate(n int) []byte {
	raw := b.bytes()
	return raw[:len(raw)-n]
}

func encodeSamplesLE(samples []int, bitDepth int) []byte {
	var buf bytes.Buffer
	for _, s := range samples {
		switch bitDepth {
		case 8:
			buf.WriteByte(uint8(s))
		case 16:
			binary.Write(&buf, binary.LittleEndian, int16(s))
		case 24:
			buf.Write(audio.Int32toInt24LEBytes(int32(s)))
		case 32:
			binary.Write(&buf, binary.LittleEndian, int32(s))
		default:
			panic(fmt.Sprintf("unsupported bit depth %d", bitDepth))
		}
	}
	return buf.Bytes()
}

// repeatFrame returns n copies of the frame (one value per channel).
func repeatFrame(frame []int, n int) []int {
	out := make([]int, 0, n*len(frame))
	for i := 0; i < n; i++ {
		out = append(out, frame...)
	}
	return out
}

func concat(slices ...[]int) []int {
	total := 0
	for _, s := range slices {
		total += len(s)
	}
	out := make([]int, 0, total)
	for _, s := range slices {
		out = append(out, s...)
	}
	return out
}

// TestWavl_PCMBuffer decodes a wavl file made of two fixed PCM segments with
// an slnt chunk in between and checks the merged, time ordered output.
func TestWavl_PCMBuffer(t *testing.T) {
	// 16 bit mono, 8kHz. Two 4-sample sound segments separated by 3 silent
	// frames holding the last sample of the first segment.
	first := []int{10, 11, 12, 13}
	second := []int{20, 21, 22, 23}
	want := concat(first, repeatFrame([]int{13}, 3), second)

	b := newWavlBuilder(1, 16, 8000)
	b.dataChunk(first)
	b.slntChunk(3)
	b.dataChunk(second)

	d := NewDecoder(bytes.NewReader(b.bytes()))
	if err := d.FwdToPCM(); err != nil {
		t.Fatalf("FwdToPCM: %v", err)
	}
	if !d.wavl {
		t.Fatal("expected the decoder to detect a wavl container")
	}
	if got := d.PCMLen(); got != int64(len(want)*2) {
		t.Fatalf("PCMLen = %d, want %d", got, len(want)*2)
	}

	// Use deliberately small buffers so reads constantly straddle the
	// data/slnt/data boundaries and partially consume frames.
	buf := &audio.IntBuffer{Data: make([]int, 3)}
	var got []int
	for {
		n, err := d.PCMBuffer(buf)
		got = append(got, buf.Data[:n]...)
		if err != nil {
			t.Fatalf("PCMBuffer: %v", err)
		}
		if n == 0 {
			break
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("decoded samples = %v, want %v", got, want)
	}
	if d.Err() != nil {
		t.Fatalf("decoder error: %v", d.Err())
	}
}

// TestWavl_FullPCMBuffer exercises the other read entry point and checks the
// reported format metadata.
func TestWavl_FullPCMBuffer(t *testing.T) {
	first := []int{100, -100, 200, -200}
	second := []int{300, -300}
	want := concat(first, repeatFrame([]int{-200}, 4), second)

	b := newWavlBuilder(1, 16, 22050)
	b.dataChunk(first)
	b.slntChunk(4)
	b.dataChunk(second)

	d := NewDecoder(bytes.NewReader(b.bytes()))
	full, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatalf("FullPCMBuffer: %v", err)
	}
	if !reflect.DeepEqual(full.Data, want) {
		t.Fatalf("FullPCMBuffer = %v, want %v", full.Data, want)
	}
	if full.Format.NumChannels != 1 || full.Format.SampleRate != 22050 {
		t.Fatalf("unexpected format: %+v", full.Format)
	}
	if full.SourceBitDepth != 16 {
		t.Fatalf("SourceBitDepth = %d, want 16", full.SourceBitDepth)
	}
	if full.NumFrames() != len(want) {
		t.Fatalf("NumFrames = %d, want %d", full.NumFrames(), len(want))
	}
	// A subsequent read must report clean EOF.
	buf := &audio.IntBuffer{Data: make([]int, 8)}
	if n, err := d.PCMBuffer(buf); err != nil || n != 0 {
		t.Fatalf("post-exhaustion read = (%d, %v), want (0, nil)", n, err)
	}
}

// TestWavl_Stereo verifies silence holds the previous value independently for
// every channel and the frame accounting stays correct.
func TestWavl_Stereo(t *testing.T) {
	// stereo 24 bit, frames of 2 samples each.
	firstFrames := []int{1, 2, 3, 4, 5, 6} // 3 frames, last frame {5, 6}
	secondFrames := []int{7, 8}            // 1 frame
	want := concat(firstFrames, repeatFrame([]int{5, 6}, 2), secondFrames)

	b := newWavlBuilder(2, 24, 48000)
	b.dataChunk(firstFrames)
	b.slntChunk(2)
	b.dataChunk(secondFrames)

	d := NewDecoder(bytes.NewReader(b.bytes()))
	full, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatalf("FullPCMBuffer: %v", err)
	}
	if !reflect.DeepEqual(full.Data, want) {
		t.Fatalf("stereo decode = %v, want %v", full.Data, want)
	}
	if full.NumFrames() != 6 {
		t.Fatalf("NumFrames = %d, want 6", full.NumFrames())
	}
}

// TestWavl_MultipleDataAndPadding covers multiple data chunks, an odd sized
// data chunk with a padding byte, an 8 bit unsigned segment and unknown chunks
// interleaved in the sequence.
func TestWavl_MultipleDataAndPadding(t *testing.T) {
	// 8 bit mono, frame size 1: an odd data size exercises word padding.
	a := []int{10, 11, 12} // 3 bytes, odd -> one padding byte on disk
	c := []int{40, 41}
	want := concat(a, repeatFrame([]int{12}, 2), c)

	b := newWavlBuilder(1, 8, 16000)
	b.unknownChunk([4]byte{'J', 'U', 'N', 'K'}, []byte{0xAA, 0xBB})
	b.dataChunk(a)
	b.unknownChunk([4]byte{'f', 'a', 'c', 't'}, []byte{0, 0, 0, 0})
	b.slntChunk(2)
	b.dataChunk(c)
	b.unknownChunk([4]byte{'L', 'I', 'S', 'T'}, []byte{'I', 'N', 'F', 'O'})

	raw := b.bytes()
	// Sanity check the odd chunk really got padded.
	expectedPadded := []byte{'d', 'a', 't', 'a', 0x03, 0x00, 0x00, 0x00, 10, 11, 12, 0}
	if !bytes.Contains(raw, expectedPadded) {
		t.Fatal("test fixture missing expected padded data chunk")
	}

	d := NewDecoder(bytes.NewReader(raw))
	buf := &audio.IntBuffer{Data: make([]int, 2)} // tiny reads cross every boundary
	var got []int
	for {
		n, err := d.PCMBuffer(buf)
		if err != nil {
			t.Fatalf("PCMBuffer: %v", err)
		}
		got = append(got, buf.Data[:n]...)
		if n == 0 {
			break
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("decoded = %v, want %v", got, want)
	}
}

// TestWavl_LeadingAndTrailingSlnt checks a leading silence (no previous sample
// => zeroes) and a trailing silence (holds the last frame).
func TestWavl_LeadingAndTrailingSlnt(t *testing.T) {
	sound := []int{5, 6, 7}
	want := concat(repeatFrame([]int{0}, 2), sound, repeatFrame([]int{7}, 2))

	b := newWavlBuilder(1, 16, 8000)
	b.slntChunk(2)
	b.dataChunk(sound)
	b.slntChunk(2)

	d := NewDecoder(bytes.NewReader(b.bytes()))
	full, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatalf("FullPCMBuffer: %v", err)
	}
	if !reflect.DeepEqual(full.Data, want) {
		t.Fatalf("decoded = %v, want %v", full.Data, want)
	}
}

// TestWavl_RawByteStream verifies the merging reader itself against a golden
// byte stream, including sub-frame sized reads straddling a slnt boundary.
func TestWavl_RawByteStream(t *testing.T) {
	first := []int{0x0102, 0x0304} // little-endian: 02 01 04 03
	second := []int{0x0506}
	want := encodeSamplesLE(concat(first, repeatFrame([]int{0x0304}, 2), second), 16)

	b := newWavlBuilder(1, 16, 8000)
	b.dataChunk(first)
	b.slntChunk(2)
	b.dataChunk(second)

	d := NewDecoder(bytes.NewReader(b.bytes()))
	if err := d.FwdToPCM(); err != nil {
		t.Fatal(err)
	}
	r := d.pcmReader()
	var got []byte
	small := make([]byte, 3) // smaller than a frame, crosses every boundary
	for {
		n, err := r.Read(small)
		got = append(got, small[:n]...)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("stream read: %v", err)
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("byte stream = % x, want % x", got, want)
	}
}

// TestWavl_Rewind ensures the whole sequence can be re-read from the start via
// the existing Rewind entry point.
func TestWavl_Rewind(t *testing.T) {
	first := []int{1, 2, 3}
	second := []int{9, 8}
	want := concat(first, repeatFrame([]int{3}, 2), second)

	b := newWavlBuilder(1, 16, 8000)
	b.dataChunk(first)
	b.slntChunk(2)
	b.dataChunk(second)

	d := NewDecoder(bytes.NewReader(b.bytes()))
	firstRead, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Rewind(); err != nil {
		t.Fatalf("Rewind: %v", err)
	}
	secondRead, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatalf("read after rewind: %v", err)
	}
	if !reflect.DeepEqual(firstRead.Data, secondRead.Data) || !reflect.DeepEqual(secondRead.Data, want) {
		t.Fatalf("rewind mismatch\nfirst  = %v\nsecond = %v\nwant   = %v", firstRead.Data, secondRead.Data, want)
	}
}

// TestWavl_Duration checks the silence is accounted for in the duration.
func TestWavl_Duration(t *testing.T) {
	// 16 bit mono @ 8000Hz => 2 bytes/frame.
	// 4 sound frames + 4 silence frames + 2 sound frames = 10 frames.
	b := newWavlBuilder(1, 16, 8000)
	b.dataChunk([]int{1, 2, 3, 4})
	b.slntChunk(4)
	b.dataChunk([]int{5, 6})

	d := NewDecoder(bytes.NewReader(b.bytes()))
	if err := d.FwdToPCM(); err != nil {
		t.Fatal(err)
	}
	dur, err := d.Duration()
	if err != nil {
		t.Fatal(err)
	}
	// 10 frames / 8000 Hz = 1.25ms
	if want := 1250 * 1000; int(dur.Nanoseconds()) != want {
		t.Fatalf("duration = %v, want %d ns", dur, want)
	}
}

// TestWavl_Errors exercises the deterministic failure modes.
func TestWavl_Errors(t *testing.T) {
	goodFirst := []int{1, 2, 3}
	goodSecond := []int{4, 5}

	t.Run("truncated data chunk", func(t *testing.T) {
		b := newWavlBuilder(1, 16, 8000)
		b.dataChunk(goodFirst)
		b.slntChunk(2)
		b.dataChunk(goodSecond)
		// Drop the final 2 payload bytes of the last data chunk.
		raw := b.truncate(2)
		d := NewDecoder(bytes.NewReader(raw))
		if err := d.FwdToPCM(); err != nil {
			t.Fatalf("scan should not fail, got %v", err)
		}
		_, err := d.FullPCMBuffer()
		if !errors.Is(err, ErrWavlTruncated) {
			t.Fatalf("err = %v, want ErrWavlTruncated", err)
		}
	})

	t.Run("slnt wrong size", func(t *testing.T) {
		b := newWavlBuilder(1, 16, 8000)
		b.dataChunk(goodFirst)
		b.rawSlntChunk([]byte{1, 2, 3}) // 3 bytes instead of 4
		b.dataChunk(goodSecond)
		d := NewDecoder(bytes.NewReader(b.bytes()))
		err := d.FwdToPCM()
		if !errors.Is(err, ErrWavlSlntSize) {
			t.Fatalf("err = %v, want ErrWavlSlntSize", err)
		}
	})

	t.Run("slnt zero count", func(t *testing.T) {
		b := newWavlBuilder(1, 16, 8000)
		b.dataChunk(goodFirst)
		b.slntChunk(0)
		b.dataChunk(goodSecond)
		d := NewDecoder(bytes.NewReader(b.bytes()))
		err := d.FwdToPCM()
		if !errors.Is(err, ErrWavlSlntCount) {
			t.Fatalf("err = %v, want ErrWavlSlntCount", err)
		}
	})

	t.Run("slnt oversized count", func(t *testing.T) {
		b := newWavlBuilder(1, 16, 8000)
		b.dataChunk(goodFirst)
		// 2 bytes per frame; this count would synthesize well over the
		// 256 GiB decoded-size guard.
		payload := make([]byte, 4)
		binary.LittleEndian.PutUint32(payload, uint32(maxWavlPCMBytes/2))
		b.rawSlntChunk(payload)
		b.dataChunk(goodSecond)
		d := NewDecoder(bytes.NewReader(b.bytes()))
		err := d.FwdToPCM()
		if !errors.Is(err, ErrWavlSlntCount) {
			t.Fatalf("err = %v, want ErrWavlSlntCount", err)
		}
	})

	t.Run("slnt cumulative size over cap", func(t *testing.T) {
		// Two individually valid counts whose combined output crosses the cap.
		b := newWavlBuilder(1, 16, 8000)
		b.dataChunk(goodFirst)
		payload := make([]byte, 4)
		binary.LittleEndian.PutUint32(payload, uint32(maxWavlPCMBytes/2))
		b.rawSlntChunk(payload)
		b.rawSlntChunk(payload)
		d := NewDecoder(bytes.NewReader(b.bytes()))
		err := d.FwdToPCM()
		if !errors.Is(err, ErrWavlSlntCount) {
			t.Fatalf("err = %v, want ErrWavlSlntCount", err)
		}
	})

	t.Run("no audio chunk", func(t *testing.T) {
		b := newWavlBuilder(1, 16, 8000)
		b.unknownChunk([4]byte{'J', 'U', 'N', 'K'}, []byte{1, 2, 3, 4})
		d := NewDecoder(bytes.NewReader(b.bytes()))
		err := d.FwdToPCM()
		if !errors.Is(err, ErrWavlNoData) {
			t.Fatalf("err = %v, want ErrWavlNoData", err)
		}
	})

	t.Run("non PCM format", func(t *testing.T) {
		b := newWavlBuilder(1, 16, 8000)
		b.formatTag = 3 // IEEE float, wavl is only defined for PCM
		b.dataChunk(goodFirst)
		b.slntChunk(1)
		b.dataChunk(goodSecond)
		d := NewDecoder(bytes.NewReader(b.bytes()))
		err := d.FwdToPCM()
		if !errors.Is(err, ErrWavlNotPCM) {
			t.Fatalf("err = %v, want ErrWavlNotPCM", err)
		}
	})

	t.Run("data chunk not frame aligned", func(t *testing.T) {
		// stereo 16 bit => 4 bytes per frame; 3 samples = 6 bytes is only
		// aligned on samples but not frames (6 is not a multiple of 4).
		b := newWavlBuilder(2, 16, 8000)
		b.dataChunk([]int{1, 2, 3})
		b.dataChunk([]int{4, 5, 6, 7})
		d := NewDecoder(bytes.NewReader(b.bytes()))
		err := d.FwdToPCM()
		if err == nil {
			t.Fatal("expected an error for a data chunk that is not frame aligned")
		}
	})
}

// TestWavl_RegularWaveUnaffected is a guard that a plain WAVE container still
// goes through the single-chunk path and does not get wavl treatment.
func TestWavl_RegularWaveUnaffected(t *testing.T) {
	b := newWavlBuilder(1, 16, 8000)
	b.formType = riff.WavFormatID
	b.dataChunk([]int{1, 2, 3, 4})
	d := NewDecoder(bytes.NewReader(b.bytes()))
	if err := d.FwdToPCM(); err != nil {
		t.Fatal(err)
	}
	if d.wavl {
		t.Fatal("plain WAVE file must not be treated as wavl")
	}
	if d.PCMChunk == nil || d.wavlReader != nil {
		t.Fatal("plain WAVE file must use the PCMChunk path")
	}
	full, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(full.Data, []int{1, 2, 3, 4}) {
		t.Fatalf("decoded = %v", full.Data)
	}
}

// TestWavl_BitDepths runs the same data/slnt/data scenario at every supported
// bit depth, including the unsigned 8 bit case where silence must hold the
// raw unsigned value (128 maps to zero after decoding).
func TestWavl_BitDepths(t *testing.T) {
	cases := []struct {
		bitDepth int
		first    []int
		second   []int
	}{
		{8, []int{10, 20, 30}, []int{200, 210}},
		{16, []int{-1000, 2000, -3000}, []int{4000, 5000}},
		{24, []int{-100000, 200000}, []int{300000}},
		{32, []int{-70000, 80000}, []int{90000, -95000}},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%dbit", tc.bitDepth), func(t *testing.T) {
			want := concat(tc.first, repeatFrame(tc.first[len(tc.first)-1:], 2), tc.second)
			b := newWavlBuilder(1, tc.bitDepth, 8000)
			b.dataChunk(tc.first)
			b.slntChunk(2)
			b.dataChunk(tc.second)

			d := NewDecoder(bytes.NewReader(b.bytes()))
			full, err := d.FullPCMBuffer()
			if err != nil {
				t.Fatalf("FullPCMBuffer: %v", err)
			}
			if !reflect.DeepEqual(full.Data, want) {
				t.Fatalf("%d bit decode = %v, want %v", tc.bitDepth, full.Data, want)
			}
			if full.SourceBitDepth != tc.bitDepth {
				t.Fatalf("SourceBitDepth = %d, want %d", full.SourceBitDepth, tc.bitDepth)
			}
		})
	}
}

// TestWavl_InterleavedParts chains several alternating data and slnt chunks
// with a stereo stream to make sure the frame value and counter stay coherent
// across many transitions.
func TestWavl_InterleavedParts(t *testing.T) {
	framesA := []int{1, 10}
	framesB := []int{2, 20}
	framesC := []int{3, 30}
	want := concat(
		framesA,
		repeatFrame(framesA, 2),
		framesB,
		repeatFrame(framesB, 1),
		framesC,
		repeatFrame(framesC, 3),
	)

	b := newWavlBuilder(2, 16, 12000)
	b.dataChunk(framesA)
	b.slntChunk(2)
	b.dataChunk(framesB)
	b.slntChunk(1)
	b.dataChunk(framesC)
	b.slntChunk(3)

	d := NewDecoder(bytes.NewReader(b.bytes()))
	buf := &audio.IntBuffer{Data: make([]int, 3)} // forces mid-frame reads
	var got []int
	for {
		n, err := d.PCMBuffer(buf)
		got = append(got, buf.Data[:n]...)
		if err != nil {
			t.Fatalf("PCMBuffer: %v", err)
		}
		if n == 0 {
			break
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("interleaved decode = %v, want %v", got, want)
	}
}

// TestWavl_IsValidFile makes sure the existing validation entry point accepts
// well formed wavl files.
func TestWavl_IsValidFile(t *testing.T) {
	b := newWavlBuilder(1, 16, 8000)
	b.dataChunk([]int{1, 2, 3})
	b.slntChunk(2)
	b.dataChunk([]int{4, 5})
	d := NewDecoder(bytes.NewReader(b.bytes()))
	if !d.IsValidFile() {
		t.Fatalf("expected a valid wavl file, Err: %v", d.Err())
	}
}

// TestWavl_FactAndUnknownChunks places a fact chunk (mandatory in the wavl
// recommendation) plus JUNK/unknown chunks before, between and after audio
// chunks and checks they are all ignored while ordering is preserved.
func TestWavl_FactAndUnknownChunks(t *testing.T) {
	first := []int{7, 8, 9}
	second := []int{1, 2}
	want := concat(first, repeatFrame([]int{9}, 1), second)

	b := newWavlBuilder(1, 16, 8000)
	b.unknownChunk([4]byte{'J', 'U', 'N', 'K'}, []byte{0xDE, 0xAD})
	b.dataChunk(first)
	b.unknownChunk([4]byte{'f', 'a', 'c', 't'}, []byte{1, 0, 0, 0})
	b.slntChunk(1)
	b.unknownChunk([4]byte{'X', 'Y', 'Z', 'Z'}, []byte{9, 9, 9, 9, 9})
	b.dataChunk(second)

	d := NewDecoder(bytes.NewReader(b.bytes()))
	full, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatalf("FullPCMBuffer: %v", err)
	}
	if !reflect.DeepEqual(full.Data, want) {
		t.Fatalf("decode with metadata chunks = %v, want %v", full.Data, want)
	}
}

// TestWavl_TinyReads exhaustively reads the same wavl byte stream one byte at a
// time through every possible fixed read size (1..9 bytes), which forces every
// read to straddle frame, data/slnt and slnt/data boundaries. Each run must
// produce the exact golden expansion.
func TestWavl_TinyReads(t *testing.T) {
	// stereo 16 bit: frame size 4; silence repeats the last frame {50, 60}.
	first := []int{10, 20, 30, 40, 50, 60} // 3 frames
	second := []int{70, 80}                // 1 frame
	want := encodeSamplesLE(concat(first, repeatFrame([]int{50, 60}, 3), second), 16)

	b := newWavlBuilder(2, 16, 8000)
	b.dataChunk(first)
	b.slntChunk(3)
	b.dataChunk(second)

	for readSize := 1; readSize <= 9; readSize++ {
		d := NewDecoder(bytes.NewReader(b.bytes()))
		if err := d.FwdToPCM(); err != nil {
			t.Fatalf("readSize %d: FwdToPCM: %v", readSize, err)
		}
		r := d.pcmReader()
		tmp := make([]byte, readSize)
		var got []byte
		for {
			n, err := r.Read(tmp)
			got = append(got, tmp[:n]...)
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatalf("readSize %d: %v", readSize, err)
			}
			if n == 0 {
				t.Fatalf("readSize %d: reader returned 0, nil", readSize)
			}
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("readSize %d:\n got % x\nwant % x", readSize, got, want)
		}
	}
}

// TestWavl_ReadMetadata ensures the metadata pass walks a wavl container
// without choking on data/slnt chunks (they are simply drained), while the
// decoder remains usable after a rewind.
func TestWavl_ReadMetadata(t *testing.T) {
	first := []int{1, 2, 3}
	second := []int{8, 9}
	want := concat(first, repeatFrame([]int{3}, 2), second)

	b := newWavlBuilder(1, 16, 8000)
	// INFO LIST before the audio, plus an unknown chunk in the sequence.
	listPayload := []byte("INFO")
	listPayload = append(listPayload, 'I', 'N', 'A', 'M')
	listPayload = append(listPayload, encodeLE32(6)...)
	listPayload = append(listPayload, 't', 'i', 't', 'l', 'e', 0)
	b.chunk(CIDList, listPayload)
	b.dataChunk(first)
	b.slntChunk(2)
	b.unknownChunk([4]byte{'J', 'U', 'N', 'K'}, []byte{1, 2})
	b.dataChunk(second)

	d := NewDecoder(bytes.NewReader(b.bytes()))
	d.ReadMetadata()
	if err := d.Err(); err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	if d.Metadata == nil || d.Metadata.Title != "title" {
		t.Fatalf("metadata not parsed: %+v", d.Metadata)
	}

	// Metadata drained the reader; rewind and decode the full PCM stream.
	if err := d.Rewind(); err != nil {
		t.Fatalf("Rewind: %v", err)
	}
	full, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatalf("FullPCMBuffer after metadata+rewind: %v", err)
	}
	if !reflect.DeepEqual(full.Data, want) {
		t.Fatalf("decode after metadata = %v, want %v", full.Data, want)
	}
}

// encodeLE32 is a test helper for little-endian uint32 payloads.
func encodeLE32(v uint32) []byte {
	out := make([]byte, 4)
	binary.LittleEndian.PutUint32(out, v)
	return out
}
