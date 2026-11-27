package wav

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"reflect"
	"testing"

	"github.com/go-audio/audio"
	"github.com/go-audio/riff"
)

// chunkReader hands out the wrapped data in chunks of the given sizes, cycling
// through the pattern. It deliberately implements only io.Reader so the tests
// prove the streaming decoder never needs to seek.
type chunkReader struct {
	data    []byte
	off     int
	sizes   []int
	sizeIdx int
	reads   int
}

func (r *chunkReader) Read(p []byte) (int, error) {
	if r.off >= len(r.data) {
		return 0, io.EOF
	}
	r.reads++
	size := len(p)
	if len(r.sizes) > 0 {
		size = r.sizes[r.sizeIdx%len(r.sizes)]
		r.sizeIdx++
	}
	if size > len(p) {
		size = len(p)
	}
	if size > len(r.data)-r.off {
		size = len(r.data) - r.off
	}
	copy(p, r.data[r.off:r.off+size])
	r.off += size
	return size, nil
}

func le16(v uint16) []byte {
	b := make([]byte, 2)
	binary.LittleEndian.PutUint16(b, v)
	return b
}

func le32(v uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	return b
}

func wavChunk(id string, payload []byte) []byte {
	if len(id) != 4 {
		panic("chunk id must be 4 bytes")
	}
	out := append([]byte(id), le32(uint32(len(payload)))...)
	out = append(out, payload...)
	if len(payload)%2 == 1 {
		out = append(out, 0)
	}
	return out
}

func fmtChunk(formatTag, chans uint16, rate uint32, bits uint16) []byte {
	align := uint16(int(chans) * int((bits+7)/8))
	return wavChunk("fmt ", concat(
		le16(formatTag),
		le16(chans),
		le32(rate),
		le32(rate*uint32(align)),
		le16(align),
		le16(bits),
	))
}

func concat(parts ...[]byte) []byte {
	var out []byte
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// wavContainer wraps chunks into a RIFF/WAVE container with a correct size.
func wavContainer(chunks ...[]byte) []byte {
	body := concat(chunks...)
	out := append([]byte("RIFF"), le32(uint32(len(body)+4))...)
	out = append(out, []byte("WAVE")...)
	out = append(out, body...)
	return out
}

func pcm16(samples ...int) []byte {
	out := make([]byte, 0, len(samples)*2)
	for _, s := range samples {
		out = append(out, le16(uint16(int16(s)))...)
	}
	return out
}

func decodeStream(t *testing.T, r io.Reader, bufSamples int) ([]int, error) {
	t.Helper()
	d := NewStreamDecoder(r)
	var samples []int
	buf := &audio.IntBuffer{Data: make([]int, bufSamples)}
	for {
		n, err := d.PCMBuffer(buf)
		samples = append(samples, buf.Data[:n]...)
		if err != nil {
			return samples, err
		}
		if n == 0 {
			return samples, d.Err()
		}
	}
}

func simpleStereoWAV(samples []int) []byte {
	data := make([]byte, 0, len(samples)*2)
	for _, s := range samples {
		data = append(data, le16(uint16(int16(s)))...)
	}
	return wavContainer(
		fmtChunk(1, 1, 8000, 16),
		wavChunk("data", data),
	)
}

// pcm16StereoWAV builds a 16 bit stereo stream whose channels are interleaved
// as the caller provides them.
func pcm16StereoWAV(samples []int) []byte {
	return wavContainer(
		fmtChunk(1, 2, 8000, 16),
		wavChunk("data", pcm16(samples...)),
	)
}

// TestStreamDecoderStereoInterleaving pins the channel order for a stereo
// stream: sample n of channel 0 must stay before sample n of channel 1.
func TestStreamDecoderStereoInterleaving(t *testing.T) {
	want := []int{10, -10, 20, -20, 30, -30}
	stream := pcm16StereoWAV(want)
	for _, sizes := range [][]int{{1}, {3, 5, 1}} {
		got, err := decodeStream(t, &chunkReader{data: stream, sizes: sizes}, 4)
		if err != nil {
			t.Fatalf("splits=%v: %v", sizes, err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("splits=%v: samples = %v, want %v", sizes, got, want)
		}
	}
	d := NewStreamDecoder(&chunkReader{data: stream, sizes: []int{2}})
	buf := &audio.IntBuffer{Data: make([]int, 2)}
	n, err := d.PCMBuffer(buf)
	if err != nil || n != 2 {
		t.Fatalf("PCMBuffer = %d, %v", n, err)
	}
	if buf.Format == nil || buf.Format.NumChannels != 2 {
		t.Fatalf("format = %+v, want 2 channels", buf.Format)
	}
	if buf.NumFrames() != 1 {
		t.Fatalf("NumFrames = %d, want 1", buf.NumFrames())
	}
}

// TestStreamDecoderFixedSamples is the fixed [1,-2,3] check: the same bytes
// must decode to the same samples regardless of how the reader splits them.
func TestStreamDecoderFixedSamples(t *testing.T) {
	want := []int{1, -2, 3}
	wantBytes := []byte{0x01, 0x00, 0xfe, 0xff, 0x03, 0x00}
	if got := pcm16(want...); !bytes.Equal(got, wantBytes) {
		t.Fatalf("test fixture bytes = % x, want % x", got, wantBytes)
	}
	streamBytes := simpleStereoWAV(want)

	splitPatterns := map[string][]int{
		"single byte":       {1},
		"two bytes":         {2},
		"three bytes":       {3},
		"uneven":            {3, 1, 4, 1, 5, 9, 2, 6},
		"header sized":      {4, 8, 12},
		"whole":             {1 << 16},
		"one then the rest": {1, 1 << 16},
	}

	for name, sizes := range splitPatterns {
		t.Run(name, func(t *testing.T) {
			for _, bufSamples := range []int{1, 2, 3, 7} {
				info := NewStreamDecoder(&chunkReader{data: streamBytes, sizes: sizes})
				info.ReadInfo()
				if info.NumChans != 1 || info.BitDepth != 16 || info.SampleRate != 8000 {
					t.Fatalf("fmt not parsed: %d chans, %d bits, %d Hz", info.NumChans, info.BitDepth, info.SampleRate)
				}
				got, err := decodeStream(t, &chunkReader{data: streamBytes, sizes: sizes}, bufSamples)
				if err != nil {
					t.Fatalf("buf=%d: unexpected error %v", bufSamples, err)
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("buf=%d: samples = %v, want %v", bufSamples, got, want)
				}
			}

			// FullPCMBuffer must return the same samples.
			r := &chunkReader{data: streamBytes, sizes: sizes}
			buf, err := NewStreamDecoder(r).FullPCMBuffer()
			if err != nil {
				t.Fatalf("FullPCMBuffer: %v", err)
			}
			if !reflect.DeepEqual(buf.Data, want) {
				t.Fatalf("FullPCMBuffer samples = %v, want %v", buf.Data, want)
			}
			if buf.SourceBitDepth != 16 {
				t.Fatalf("SourceBitDepth = %d, want 16", buf.SourceBitDepth)
			}
			if buf.Format == nil || buf.Format.NumChannels != 1 || buf.Format.SampleRate != 8000 {
				t.Fatalf("unexpected format %+v", buf.Format)
			}
		})
	}
}

// TestStreamDecoderMatchesSeekableDecoder checks that every fixture decodes to
// exactly the same samples as the seekable decoder.
func TestStreamDecoderMatchesSeekableDecoder(t *testing.T) {
	fixtures := []string{
		"fixtures/kick.wav",
		"fixtures/kick-16b441k.wav",
		"fixtures/bass.wav",
		"fixtures/8bit.wav",
		"fixtures/32bit.wav",
		"fixtures/flloop.wav",
		"fixtures/listChunkInHeader.wav",
		"fixtures/listinfo.wav",
		"fixtures/dirty-kick-24b441k.wav",
		"fixtures/padded24b.wav",
		"fixtures/misaligned-chunk.wav",
		"fixtures/bwf.wav",
	}
	for _, name := range fixtures {
		name := name
		t.Run(name, func(t *testing.T) {
			f, err := os.Open(name)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()

			// PCMBuffer is the reference sample semantics: it drops a partial
			// sample left at the end of the data chunk.
			wantSamples, wantFormat, wantDepth := collectDecoderSamples(t, NewDecoder(f))

			raw, err := ioutil.ReadFile(name)
			if err != nil {
				t.Fatal(err)
			}
			// Cycle through uneven split sizes to exercise packet boundaries.
			r := &chunkReader{data: raw, sizes: []int{1, 3, 7, 2, 11, 5}}
			gotSamples, gotFormat, gotDepth := collectStreamSamples(t, NewStreamDecoder(r))
			if len(gotSamples) != len(wantSamples) {
				t.Fatalf("decoded %d samples, seekable decoder decoded %d", len(gotSamples), len(wantSamples))
			}
			if !reflect.DeepEqual(gotSamples, wantSamples) {
				for i := range wantSamples {
					if gotSamples[i] != wantSamples[i] {
						t.Fatalf("sample %d: stream=%d seekable=%d", i, gotSamples[i], wantSamples[i])
					}
				}
			}
			if gotFormat.NumChannels != wantFormat.NumChannels ||
				gotFormat.SampleRate != wantFormat.SampleRate ||
				gotDepth != wantDepth {
				t.Fatalf("format mismatch: stream=%+v/%d seekable=%+v/%d",
					gotFormat, gotDepth, wantFormat, wantDepth)
			}
		})
	}
}

// collectDecoderSamples drains a seekable decoder through PCMBuffer.
func collectDecoderSamples(t *testing.T, d *Decoder) ([]int, *audio.Format, int) {
	t.Helper()
	var samples []int
	buf := &audio.IntBuffer{Data: make([]int, 256)}
	for {
		n, err := d.PCMBuffer(buf)
		if err != nil {
			t.Fatalf("seekable PCMBuffer: %v", err)
		}
		samples = append(samples, buf.Data[:n]...)
		if n == 0 {
			break
		}
	}
	if buf.Format == nil {
		t.Fatal("seekable decoder returned no format")
	}
	return samples, buf.Format, buf.SourceBitDepth
}

// collectStreamSamples drains a streaming decoder through PCMBuffer.
func collectStreamSamples(t *testing.T, d *StreamDecoder) ([]int, *audio.Format, int) {
	t.Helper()
	var samples []int
	buf := &audio.IntBuffer{Data: make([]int, 256)}
	for {
		n, err := d.PCMBuffer(buf)
		if err != nil {
			t.Fatalf("stream PCMBuffer: %v", err)
		}
		samples = append(samples, buf.Data[:n]...)
		if n == 0 {
			break
		}
	}
	if buf.Format == nil {
		t.Fatal("stream decoder returned no format")
	}
	return samples, buf.Format, buf.SourceBitDepth
}

// TestStreamDecoderInterleavingAndBitDepths checks the sample values for every
// supported width, including a stereo stream whose channels must stay
// interleaved, under arbitrary reader splits.
func TestStreamDecoderInterleavingAndBitDepths(t *testing.T) {
	tests := []struct {
		name    string
		bits    uint16
		chans   uint16
		payload []byte
		want    []int
		splits  [][]int
	}{
		{
			name:    "16 bit stereo interleaved",
			bits:    16,
			chans:   2,
			payload: pcm16(100, -100, 200, -200, 300, -300),
			want:    []int{100, -100, 200, -200, 300, -300},
		},
		{
			name:    "8 bit mono is unsigned",
			bits:    8,
			chans:   1,
			payload: []byte{0, 1, 127, 128, 255},
			want:    []int{0, 1, 127, 128, 255},
		},
		{
			name:    "24 bit stereo",
			bits:    24,
			chans:   2,
			payload: concat(audio.Int32toInt24LEBytes(1), audio.Int32toInt24LEBytes(-1), audio.Int32toInt24LEBytes(0x7fffff), audio.Int32toInt24LEBytes(-0x800000)),
			want:    []int{1, -1, 0x7fffff, -0x800000},
		},
		{
			name:    "32 bit mono",
			bits:    32,
			chans:   1,
			payload: concat(le32(1), le32(0xffffffff), le32(0x7fffffff), le32(0x80000000)),
			want:    []int{1, -1, 0x7fffffff, -0x80000000},
		},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			stream := wavContainer(fmtChunk(1, tc.chans, 8000, tc.bits), wavChunk("data", tc.payload))
			for _, sizes := range [][]int{{1}, {2}, {3}, {5, 1, 7}, {1 << 16}} {
				for _, bufSamples := range []int{1, 2, 5, 100} {
					got, err := decodeStream(t, &chunkReader{data: stream, sizes: sizes}, bufSamples)
					if err != nil {
						t.Fatalf("splits=%v buf=%d: %v", sizes, bufSamples, err)
					}
					if !reflect.DeepEqual(got, tc.want) {
						t.Fatalf("splits=%v buf=%d: samples = %v, want %v", sizes, bufSamples, got, tc.want)
					}
				}
			}
		})
	}
}

// TestStreamDecoderSplitInvariance feeds the exact same container through many
// reader split patterns and requires identical output every time.
func TestStreamDecoderSplitInvariance(t *testing.T) {
	raw, err := ioutil.ReadFile("fixtures/bass.wav")
	if err != nil {
		t.Fatal(err)
	}
	reference, _, _ := collectStreamSamples(t, NewStreamDecoder(&chunkReader{data: raw, sizes: []int{1 << 20}}))
	patterns := [][]int{
		{1},
		{2},
		{3},
		{5},
		{4096},
		{1, 2, 3, 4, 5, 6, 7, 8, 9},
		{6, 1, 1, 6},
		{7, 13, 1, 4093},
	}
	for _, sizes := range patterns {
		got, _, _ := collectStreamSamples(t, NewStreamDecoder(&chunkReader{data: raw, sizes: sizes}))
		if len(got) != len(reference) {
			t.Fatalf("splits=%v: %d samples, want %d", sizes, len(got), len(reference))
		}
		for i := range reference {
			if got[i] != reference[i] {
				t.Fatalf("splits=%v: sample %d = %d, want %d", sizes, i, got[i], reference[i])
			}
		}
	}
}

func TestStreamDecoderDurationMatchesSeekableDecoder(t *testing.T) {
	for _, name := range []string{"fixtures/kick.wav", "fixtures/kick-16b441k.wav", "fixtures/bass.wav"} {
		f, err := os.Open(name)
		if err != nil {
			t.Fatal(err)
		}
		want, err := NewDecoder(f).Duration()
		f.Close()
		if err != nil {
			t.Fatalf("%s: seekable duration: %v", name, err)
		}
		raw, err := ioutil.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		got, err := NewStreamDecoder(&chunkReader{data: raw, sizes: []int{3}}).Duration()
		if err != nil {
			t.Fatalf("%s: stream duration: %v", name, err)
		}
		if got != want {
			t.Fatalf("%s: stream duration = %s, seekable = %s", name, got, want)
		}
	}
}

// TestStreamDecoderRandomizedSplits cross-checks the streaming decoder
// against the seekable decoder with pseudo-random reader splits and buffer
// sizes.
func TestStreamDecoderRandomizedSplits(t *testing.T) {
	raw, err := ioutil.ReadFile("fixtures/bass.wav")
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open("fixtures/bass.wav")
	if err != nil {
		t.Fatal(err)
	}
	want, _, _ := collectDecoderSamples(t, NewDecoder(f))
	f.Close()

	rng := rand.New(rand.NewSource(42))
	for iter := 0; iter < 200; iter++ {
		sizes := make([]int, 1+rng.Intn(8))
		for i := range sizes {
			sizes[i] = 1 + rng.Intn(5000)
		}
		bufSamples := 1 + rng.Intn(300)
		got, err := decodeStream(t, &chunkReader{data: raw, sizes: sizes}, bufSamples)
		if err != nil {
			t.Fatalf("iter %d splits=%v buf=%d: %v", iter, sizes, bufSamples, err)
		}
		if len(got) != len(want) {
			t.Fatalf("iter %d: %d samples, want %d", iter, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("iter %d splits=%v buf=%d: sample %d = %d, want %d", iter, sizes, bufSamples, i, got[i], want[i])
			}
		}
	}
}

// TestStreamDecoderNoPreRead verifies headers are parsed without consuming the
// PCM payload: playback can start before the stream completes.
func TestStreamDecoderNoPreRead(t *testing.T) {
	raw, err := ioutil.ReadFile("fixtures/kick.wav")
	if err != nil {
		t.Fatal(err)
	}
	r := &chunkReader{data: raw, sizes: []int{5}}
	d := NewStreamDecoder(r)
	d.ReadInfo()
	if d.NumChans != 1 || d.BitDepth != 16 {
		t.Fatalf("ReadInfo didn't populate the format: %+v", d)
	}
	// fmt chunk ends at byte 36, the data header is 8 bytes -> data starts at 44.
	if r.off > 44 {
		t.Fatalf("ReadInfo consumed %d bytes, PCM data starts at 44", r.off)
	}
	if r.off == len(raw) {
		t.Fatal("ReadInfo consumed the whole file")
	}

	// Reading a single frame must not require the rest of the payload.
	buf := &audio.IntBuffer{Data: make([]int, 1)}
	n, err := d.PCMBuffer(buf)
	if err != nil || n != 1 {
		t.Fatalf("PCMBuffer = %d, %v", n, err)
	}
	if r.off > 44+16 {
		t.Fatalf("read %d bytes for a single 16-bit sample, expected at most 46", r.off+8)
	}
}

// TestStreamDecoderOddChunkPadding ensures odd sized unknown chunks and the
// data chunk's padding byte don't shift the PCM samples.
func TestStreamDecoderOddChunkPadding(t *testing.T) {
	// 8 bit mono: samples are single bytes so the data chunk size is odd.
	payload := []byte{1, 2, 3}
	stream := wavContainer(
		wavChunk("JUNK", []byte{0xaa, 0xbb, 0xcc}), // odd unknown chunk + pad
		fmtChunk(1, 1, 8000, 8),
		wavChunk("LIST", []byte{1, 2, 3, 4, 5}), // odd unknown chunk + pad
		wavChunk("data", payload),
	)
	// wavChunk appends the padding byte; the data header must still advertise 3.
	if !bytes.Contains(stream, concat([]byte("data"), le32(3))) {
		t.Fatal("test fixture doesn't advertise an odd data chunk size")
	}

	got, err := decodeStream(t, &chunkReader{data: stream, sizes: []int{1}}, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(got, []int{1, 2, 3}) {
		t.Fatalf("samples = %v, want [1 2 3]", got)
	}
	d := NewStreamDecoder(&chunkReader{data: stream, sizes: []int{1}})
	buf, err := d.FullPCMBuffer()
	if err != nil || !reflect.DeepEqual(buf.Data, []int{1, 2, 3}) {
		t.Fatalf("FullPCMBuffer = %v, %v", buf.Data, err)
	}
	if d.PCMLen() != 3 {
		t.Fatalf("PCMLen = %d, want 3", d.PCMLen())
	}
}

// TestStreamDecoderUnknownDataSize covers the 0xFFFFFFFF "size not known yet"
// marker used by streaming writers: the payload is read until EOF.
func TestStreamDecoderUnknownDataSize(t *testing.T) {
	data := pcm16(7, -8, 9)
	stream := wavContainer(
		fmtChunk(1, 1, 8000, 16),
		append([]byte("data"), le32(0xffffffff)...),
		data,
	)
	d := NewStreamDecoder(&chunkReader{data: stream, sizes: []int{1, 2, 3}})
	buf, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatalf("FullPCMBuffer: %v", err)
	}
	if !reflect.DeepEqual(buf.Data, []int{7, -8, 9}) {
		t.Fatalf("samples = %v", buf.Data)
	}
	if d.PCMLen() != -1 {
		t.Fatalf("PCMLen = %d, want -1 for an unknown size", d.PCMLen())
	}
	if _, err := d.Duration(); err == nil {
		t.Fatal("expected an error computing the duration of an unbounded stream")
	}
}

func TestStreamDecoderErrors(t *testing.T) {
	validFmt := fmtChunk(1, 1, 8000, 16)
	tests := []struct {
		name   string
		stream []byte
		errIs  []error
	}{
		{
			name:   "not a RIFF stream",
			stream: []byte("NOPE\x00\x00\x00\x00WAVE"),
			errIs:  []error{ErrNotWAVStream},
		},
		{
			name:   "unsupported form type",
			stream: []byte("RIFF\x00\x00\x00\x00AVI "),
			errIs:  []error{ErrUnsupportedStreamFormat},
		},
		{
			name:   "truncated container header",
			stream: []byte("RIFF\x00\x00"),
			errIs:  []error{ErrStreamTruncated, io.ErrUnexpectedEOF},
		},
		{
			name:   "fmt chunk too small",
			stream: wavContainer(wavChunk("fmt ", make([]byte, 10))),
			errIs:  []error{ErrInvalidStreamChunkSize},
		},
		{
			name:   "fmt chunk missing",
			stream: wavContainer(wavChunk("data", pcm16(1, 2))),
			errIs:  []error{ErrStreamFmtMissing},
		},
		{
			name:   "float format unsupported",
			stream: wavContainer(fmtChunk(3, 1, 8000, 32), wavChunk("data", make([]byte, 8))),
			errIs:  []error{ErrUnsupportedStreamFormat},
		},
		{
			name:   "compressed format unsupported",
			stream: wavContainer(fmtChunk(6, 1, 8000, 8), wavChunk("data", make([]byte, 8))),
			errIs:  []error{ErrUnsupportedStreamFormat},
		},
		{
			name:   "unusual bit depth unsupported",
			stream: wavContainer(fmtChunk(1, 1, 8000, 12), wavChunk("data", make([]byte, 8))),
			errIs:  []error{ErrUnsupportedStreamFormat},
		},
		{
			name: "zero channels",
			stream: wavContainer(wavChunk("fmt ", concat(
				le16(1), le16(0), le32(8000), le32(0), le16(0), le16(16),
			)), wavChunk("data", make([]byte, 8))),
			errIs: []error{ErrInvalidStreamFmt},
		},
		{
			name:   "truncated data payload",
			stream: concat(wavContainer(validFmt, append([]byte("data"), le32(100)...)), pcm16(1, 2)),
			errIs:  []error{ErrStreamTruncated, io.ErrUnexpectedEOF},
		},
		{
			name: "unknown size marker on a skipped chunk",
			// JUNK with the 0xFFFFFFFF marker can't be skipped safely.
			stream: wavContainer(append([]byte("JUNK"), le32(0xffffffff)...), validFmt, wavChunk("data", pcm16(1))),
			errIs:  []error{ErrInvalidStreamChunkSize},
		},
		{
			name:   "no data chunk",
			stream: wavContainer(validFmt),
			errIs:  []error{ErrPCMChunkNotFound},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			d := NewStreamDecoder(&chunkReader{data: tc.stream, sizes: []int{1}})
			_, err := d.FullPCMBuffer()
			if err == nil {
				t.Fatalf("expected an error, got none (decoder err=%v)", d.Err())
			}
			for _, want := range tc.errIs {
				if !errors.Is(err, want) {
					t.Fatalf("error %v doesn't match %v", err, want)
				}
			}
			if errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("unexpected unadorned io.EOF: %v", err)
			}
		})
	}
}

func TestStreamDecoderPartialDataIsNotPreRead(t *testing.T) {
	// A data chunk declared with more payload than the fixture contains must
	// yield the readable samples followed by an explicit truncation error.
	stream := concat(
		wavContainer(fmtChunk(1, 1, 8000, 16)),
		append([]byte("data"), le32(10)...),
		pcm16(1, -2),
	)
	d := NewStreamDecoder(&chunkReader{data: stream, sizes: []int{1}})
	buf := &audio.IntBuffer{Data: make([]int, 10)}
	n, err := d.PCMBuffer(buf)
	if n != 2 {
		t.Fatalf("decoded %d samples, want the 2 available samples", n)
	}
	if !reflect.DeepEqual(buf.Data[:n], []int{1, -2}) {
		t.Fatalf("samples = %v", buf.Data[:n])
	}
	if !errors.Is(err, ErrStreamTruncated) || !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("error = %v, want a truncation error", err)
	}
}

func TestStreamDecoderNotSeekable(t *testing.T) {
	d := NewStreamDecoder(&chunkReader{data: simpleStereoWAV([]int{1})})
	if _, err := d.Seek(0, io.SeekStart); !errors.Is(err, ErrStreamNotSeekable) {
		t.Fatalf("Seek error = %v, want ErrStreamNotSeekable", err)
	}
	if err := d.Rewind(); !errors.Is(err, ErrStreamNotSeekable) {
		t.Fatalf("Rewind error = %v, want ErrStreamNotSeekable", err)
	}
}

func TestStreamDecoderAccessors(t *testing.T) {
	stream := simpleStereoWAV([]int{1, -2, 3})
	d := NewStreamDecoder(&chunkReader{data: stream, sizes: []int{2}})
	d.ReadInfo()
	if got := d.SampleBitDepth(); got != 16 {
		t.Fatalf("SampleBitDepth = %d", got)
	}
	if got := d.Format(); got == nil || got.NumChannels != 1 || got.SampleRate != 8000 {
		t.Fatalf("Format = %+v", got)
	}
	if d.WasPCMAccessed() {
		t.Fatal("PCM reported as accessed before the data chunk was reached")
	}
	if !d.IsValidFile() {
		t.Fatal("IsValidFile = false for a valid stream")
	}
	if !d.WasPCMAccessed() {
		t.Fatal("PCM not reported as accessed after IsValidFile")
	}
	if d.PCMLen() != 6 {
		t.Fatalf("PCMLen = %d", d.PCMLen())
	}
	if d.EOF() {
		t.Fatal("EOF before reading the payload")
	}
	if err := d.FwdToPCM(); err != nil {
		t.Fatalf("FwdToPCM: %v", err)
	}
	// Reading the payload advances to EOF.
	buf := &audio.IntBuffer{Data: make([]int, 3)}
	if n, err := d.PCMBuffer(buf); n != 3 || err != nil {
		t.Fatalf("PCMBuffer = %d, %v", n, err)
	}
	if n, err := d.PCMBuffer(buf); n != 0 || err != nil {
		t.Fatalf("second PCMBuffer = %d, %v", n, err)
	}
	if !d.EOF() {
		t.Fatal("EOF = false after the payload was consumed")
	}
	if err := d.Err(); err != nil {
		t.Fatalf("Err = %v after a clean EOF", err)
	}
}

func TestStreamDecoderExtensiblePCM(t *testing.T) {
	// WAVE_FORMAT_EXTENSIBLE with a PCM sub-format GUID must be accepted.
	fmtPayload := concat(
		le16(waveFormatExtensible),
		le16(1),
		le32(8000),
		le32(16000),
		le16(2),
		le16(16),
		le16(22), // cbSize
		le16(16), // valid bits per sample
		le32(0),  // channel mask
		[]byte{1, 0, 0, 0, 0, 0, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71},
	)
	stream := wavContainer(
		wavChunk("fmt ", fmtPayload),
		wavChunk("data", pcm16(5, -6)),
	)
	buf, err := NewStreamDecoder(&chunkReader{data: stream, sizes: []int{3}}).FullPCMBuffer()
	if err != nil {
		t.Fatalf("FullPCMBuffer: %v", err)
	}
	if !reflect.DeepEqual(buf.Data, []int{5, -6}) {
		t.Fatalf("samples = %v", buf.Data)
	}
}

// static read-only reader used to make sure the API works with the minimal
// io.Reader interface.
type onlyReader struct{ r io.Reader }

func (o onlyReader) Read(p []byte) (int, error) { return o.r.Read(p) }

func TestStreamDecoderAcceptsPlainReader(t *testing.T) {
	var r io.Reader = onlyReader{bytes.NewReader(simpleStereoWAV([]int{4, 5, 6}))}
	if _, ok := r.(io.Seeker); ok {
		t.Fatal("test reader unexpectedly implements io.Seeker")
	}
	buf, err := NewStreamDecoder(r).FullPCMBuffer()
	if err != nil {
		t.Fatalf("FullPCMBuffer: %v", err)
	}
	if !reflect.DeepEqual(buf.Data, []int{4, 5, 6}) {
		t.Fatalf("samples = %v", buf.Data)
	}
}

func TestStreamDecoderChunkIDs(t *testing.T) {
	// Guard against accidental ID drift: the stream parser relies on the same
	// constants as the riff parser.
	if riff.FmtID != [4]byte{'f', 'm', 't', ' '} || riff.DataFormatID != [4]byte{'d', 'a', 't', 'a'} {
		t.Fatal("unexpected riff chunk ID constants")
	}
}
