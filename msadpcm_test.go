package wav

import (
	"bytes"
	"encoding/binary"
	"io"
	"os"
	"reflect"
	"testing"

	"github.com/go-audio/audio"
)

// msADPCMStandardCoefs are the 7 prediction coefficient pairs mandated by the
// Microsoft ADPCM specification. They are stored as (coef1, coef2).
var msADPCMStandardCoefs = [][2]int16{
	{256, 0},
	{512, -256},
	{0, 0},
	{192, 64},
	{240, 0},
	{460, -208},
	{392, -232},
}

// msADPCMBlock builds a mono MS ADPCM block with the standard coefficients.
func msADPCMBlock(predictor byte, delta int16, sample1, sample2 int16, body []byte) []byte {
	b := make([]byte, 7, 7+len(body))
	b[0] = predictor
	binary.LittleEndian.PutUint16(b[1:3], uint16(delta))
	binary.LittleEndian.PutUint16(b[3:5], uint16(sample1))
	binary.LittleEndian.PutUint16(b[5:7], uint16(sample2))
	return append(b, body...)
}

// msADPCMWav assembles a WAV container with an MS ADPCM fmt chunk extension,
// an optional fact chunk (skip it when factFrames is negative) and the passed
// data chunk payload.
func msADPCMWav(numChans, blockAlign, samplesPerBlock int, coefs [][2]int16, factFrames int, data []byte) []byte {
	var fmtChunk bytes.Buffer
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(WavAudioFormatMSADPCM))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(numChans))
	binary.Write(&fmtChunk, binary.LittleEndian, uint32(22050))
	binary.Write(&fmtChunk, binary.LittleEndian, uint32(22050*blockAlign/samplesPerBlock))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(blockAlign))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(4)) // 4 bits per sample
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(4+4*len(coefs)))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(samplesPerBlock))
	binary.Write(&fmtChunk, binary.LittleEndian, uint16(len(coefs)))
	for _, c := range coefs {
		binary.Write(&fmtChunk, binary.LittleEndian, c[0])
		binary.Write(&fmtChunk, binary.LittleEndian, c[1])
	}

	var chunks bytes.Buffer
	chunks.Write([]byte("fmt "))
	binary.Write(&chunks, binary.LittleEndian, uint32(fmtChunk.Len()))
	chunks.Write(fmtChunk.Bytes())
	if factFrames >= 0 {
		chunks.Write([]byte("fact"))
		binary.Write(&chunks, binary.LittleEndian, uint32(4))
		binary.Write(&chunks, binary.LittleEndian, uint32(factFrames))
	}
	chunks.Write([]byte("data"))
	binary.Write(&chunks, binary.LittleEndian, uint32(len(data)))
	chunks.Write(data)
	if len(data)%2 == 1 {
		chunks.WriteByte(0) // RIFF word alignment padding
	}

	var out bytes.Buffer
	out.Write([]byte("RIFF"))
	binary.Write(&out, binary.LittleEndian, uint32(4+chunks.Len()))
	out.Write([]byte("WAVE"))
	out.Write(chunks.Bytes())
	return out.Bytes()
}

func newMSADPCMDecoder(t *testing.T, wavData []byte) *Decoder {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "adpcm-*.wav")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	if _, err := f.Write(wavData); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	return NewDecoder(f)
}

// TestMSADPCMFixedBlock checks the exact expansion worked out by hand: with
// prediction coefficients (256, 0) and history samples 0 then 1000, a leading
// differential code of 0 leaves the next sample at 1000.
func TestMSADPCMFixedBlock(t *testing.T) {
	// 32 byte mono block: 7 byte header + 25 nibble bytes => 52 samples.
	block := msADPCMBlock(0, 16, 1000, 0, make([]byte, 25))
	wavData := msADPCMWav(1, 32, 52, msADPCMStandardCoefs[:1], 52, block)

	d := newMSADPCMDecoder(t, wavData)
	buf, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatal(err)
	}
	if len(buf.Data) != 52 {
		t.Fatalf("expected 52 samples, got %d", len(buf.Data))
	}
	// The two header samples are emitted sample2 then sample1.
	if buf.Data[0] != 0 || buf.Data[1] != 1000 {
		t.Fatalf("unexpected header samples: %v", buf.Data[:2])
	}
	// With coefficients (256, 0) a differential code of 0 reproduces the
	// previous sample, so the whole block stays at 1000.
	for i := 2; i < 52; i++ {
		if buf.Data[i] != 1000 {
			t.Fatalf("expected sample 1000 at position %d, got %d", i, buf.Data[i])
		}
	}
	if buf.SourceBitDepth != 16 {
		t.Fatalf("expected a 16 bit PCM source depth, got %d", buf.SourceBitDepth)
	}
	if buf.Format.NumChannels != 1 || buf.Format.SampleRate != 22050 {
		t.Fatalf("unexpected format: %+v", buf.Format)
	}
}

func TestMSADPCMExpansion(t *testing.T) {
	tests := []struct {
		name    string
		coef    [2]int16
		delta   int16
		sample1 int16
		sample2 int16
		body    []byte
		// want holds the samples following the two header samples.
		want []int
	}{
		{
			name: "coefficients (256,0) carry the sample forward",
			coef: [2]int16{256, 0}, delta: 16, sample1: 1000, sample2: 0,
			body: []byte{0x00},
			want: []int{1000, 1000},
		},
		{
			name: "differential code scales with delta",
			coef: [2]int16{256, 0}, delta: 16, sample1: 1000, sample2: 0,
			body: []byte{0x10},
			want: []int{1016, 1016},
		},
		{
			name: "negative differential code",
			coef: [2]int16{256, 0}, delta: 16, sample1: 1000, sample2: 0,
			body: []byte{0xF0},
			want: []int{984, 984},
		},
		{
			name: "delta adapts after a maximum code",
			coef: [2]int16{256, 0}, delta: 100, sample1: 1000, sample2: 0,
			// two +7 codes: the first sample is +700, the second uses the
			// adapted delta of 100*614/256 = 239.
			body: []byte{0x77},
			want: []int{1700, 3373},
		},
		{
			name: "coefficients (512,-256)",
			coef: [2]int16{512, -256}, delta: 16, sample1: 1000, sample2: 0,
			body: []byte{0x00},
			want: []int{2000, 3000},
		},
		{
			name: "clamping to the 16 bit range",
			coef: [2]int16{256, 0}, delta: 30000, sample1: 0, sample2: 0,
			body: []byte{0x70},
			want: []int{32767, 32767},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			padded := make([]byte, 25)
			copy(padded, tc.body)
			block := msADPCMBlock(0, tc.delta, tc.sample1, tc.sample2, padded)
			wavData := msADPCMWav(1, 32, 52, [][2]int16{tc.coef}, -1, block)
			d := newMSADPCMDecoder(t, wavData)
			buf, err := d.FullPCMBuffer()
			if err != nil {
				t.Fatal(err)
			}
			if len(buf.Data) != 52 {
				t.Fatalf("expected 52 samples, got %d", len(buf.Data))
			}
			for i, want := range tc.want {
				if buf.Data[2+i] != want {
					t.Fatalf("sample %d: expected %d, got %d", 2+i, want, buf.Data[2+i])
				}
			}
		})
	}
}

func TestMSADPCMInvalidBlock(t *testing.T) {
	tests := []struct {
		name      string
		block     []byte
		chans     int
		align     int
		coefCount int
	}{
		{
			name:      "predictor out of range",
			block:     msADPCMBlock(7, 16, 0, 0, make([]byte, 25)),
			chans:     1,
			align:     32,
			coefCount: len(msADPCMStandardCoefs),
		},
		{
			name:      "negative delta",
			block:     msADPCMBlock(0, -16, 0, 0, make([]byte, 25)),
			chans:     1,
			align:     32,
			coefCount: len(msADPCMStandardCoefs),
		},
		{
			name:      "zero delta",
			block:     msADPCMBlock(0, 0, 0, 0, make([]byte, 25)),
			chans:     1,
			align:     32,
			coefCount: len(msADPCMStandardCoefs),
		},
		{
			name:      "block shorter than its header",
			block:     []byte{0, 16, 0},
			chans:     1,
			align:     32,
			coefCount: len(msADPCMStandardCoefs),
		},
		{
			name: "stereo block with only one predictor",
			// A single byte can't hold the predictor for both channels.
			block:     []byte{0},
			chans:     2,
			align:     32,
			coefCount: len(msADPCMStandardCoefs),
		},
		{
			name:      "predictor index without a matching coefficient",
			block:     msADPCMBlock(1, 16, 0, 0, make([]byte, 25)),
			chans:     1,
			align:     32,
			coefCount: 1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			coefs := msADPCMStandardCoefs[:tc.coefCount]
			spb := msADPCMSamplesPerBlock(tc.align, tc.chans)
			wavData := msADPCMWav(tc.chans, tc.align, spb, coefs, -1, tc.block)
			d := newMSADPCMDecoder(t, wavData)
			if _, err := d.FullPCMBuffer(); err == nil {
				t.Fatal("expected an error for a malformed block")
			}
		})
	}
}

func TestMSADPCMTruncatedBlocks(t *testing.T) {
	full := msADPCMBlock(0, 16, 1000, 0, make([]byte, 25))
	tests := []struct {
		name string
		data []byte
		fact int
	}{
		{
			name: "last block truncated mid-body with fact",
			data: append(append([]byte{}, full...), full[:10]...),
			fact: 104,
		},
		{
			name: "block truncated below the header size",
			data: append(append([]byte{}, full...), full[:3]...),
			fact: -1,
		},
		{
			name: "fact announces more frames than the data holds",
			data: full,
			fact: 100,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			wavData := msADPCMWav(1, 32, 52, msADPCMStandardCoefs, tc.fact, tc.data)
			d := newMSADPCMDecoder(t, wavData)
			if _, err := d.FullPCMBuffer(); err == nil {
				t.Fatal("expected an error, got nil")
			}
		})
	}
}

func TestMSADPCMFactTrimsLastBlock(t *testing.T) {
	// Two full blocks worth of samples but the fact chunk only announces 60
	// frames, so the decoder must drop the trailing frames of the last block.
	full := msADPCMBlock(0, 16, 1000, 0, make([]byte, 25))
	wavData := msADPCMWav(1, 32, 52, msADPCMStandardCoefs, 60, append(append([]byte{}, full...), full...))
	d := newMSADPCMDecoder(t, wavData)
	buf, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatal(err)
	}
	if len(buf.Data) != 60 {
		t.Fatalf("expected the fact chunk to trim the output to 60 samples, got %d", len(buf.Data))
	}
}

func TestMSADPCMStereo(t *testing.T) {
	// Stereo block with a 14 byte header and 18 nibble bytes => 20 frames
	// (40 samples). The header stores predictor, delta, sample1 and sample2
	// for every channel in turn.
	block := make([]byte, 14)
	block[0] = 0                                    // left predictor
	block[1] = 0                                    // right predictor
	binary.LittleEndian.PutUint16(block[2:4], 16)   // left delta
	binary.LittleEndian.PutUint16(block[4:6], 16)   // right delta
	binary.LittleEndian.PutUint16(block[6:8], 1000) // left sample1
	binary.LittleEndian.PutUint16(block[8:10], 2000)
	binary.LittleEndian.PutUint16(block[10:12], 0) // left sample2
	binary.LittleEndian.PutUint16(block[12:14], 0)
	// First byte: high nibble (+1) is the left channel, low nibble (+3) the
	// right channel.
	block = append(block, 0x13)
	block = append(block, make([]byte, 17)...)

	wavData := msADPCMWav(2, 32, 20, msADPCMStandardCoefs, -1, block)
	d := newMSADPCMDecoder(t, wavData)
	buf, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatal(err)
	}
	// 20 frames, each with a left and right sample.
	if len(buf.Data) != 40 {
		t.Fatalf("expected 40 samples, got %d", len(buf.Data))
	}
	if buf.Format.NumChannels != 2 {
		t.Fatalf("expected 2 channels, got %d", buf.Format.NumChannels)
	}
	// Frames: (sample2 L,R), (sample1 L,R), then the first decoded frame.
	want := []int{0, 0, 1000, 2000, 1016, 2048}
	for i, w := range want {
		if buf.Data[i] != w {
			t.Fatalf("sample %d: expected %d, got %d", i, w, buf.Data[i])
		}
	}
	// Both channels must keep advancing independently.
	if buf.Data[4] == buf.Data[5] {
		t.Fatalf("expected the stereo channels to decode independently, got %v", buf.Data[4:6])
	}
}

// TestMSADPCMStreamingMatchesFull checks that PCMBuffer decoding in small
// chunks matches the single call decoder, including across block boundaries.
func TestMSADPCMStreamingMatchesFull(t *testing.T) {
	full := msADPCMBlock(0, 16, 1000, 0, make([]byte, 25))
	// Three blocks: 156 samples total, 7 more than the announced 149 frames.
	data := append(append(append([]byte{}, full...), full...), full...)
	wavData := msADPCMWav(1, 32, 52, msADPCMStandardCoefs, 149, data)

	all, err := newMSADPCMDecoder(t, wavData).FullPCMBuffer()
	if err != nil {
		t.Fatal(err)
	}

	d := newMSADPCMDecoder(t, wavData)
	var streamed []int
	buf := &audio.IntBuffer{Data: make([]int, 7)}
	for {
		n, err := d.PCMBuffer(buf)
		if err != nil {
			t.Fatal(err)
		}
		if n == 0 {
			break
		}
		streamed = append(streamed, buf.Data[:n]...)
	}
	if len(streamed) != 149 {
		t.Fatalf("expected 149 streamed samples, got %d", len(streamed))
	}
	if len(all.Data) != len(streamed) {
		t.Fatalf("full decoding returned %d samples, streaming %d", len(all.Data), len(streamed))
	}
	for i := range all.Data {
		if all.Data[i] != streamed[i] {
			t.Fatalf("sample %d differs: full=%d streaming=%d", i, all.Data[i], streamed[i])
		}
	}
}

func TestMSADPCMRewind(t *testing.T) {
	full := msADPCMBlock(0, 16, 1000, 0, make([]byte, 25))
	wavData := msADPCMWav(1, 32, 52, msADPCMStandardCoefs, 52, full)
	d := newMSADPCMDecoder(t, wavData)

	first, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Rewind(); err != nil {
		t.Fatal(err)
	}
	second, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Data) != len(second.Data) {
		t.Fatalf("expected %d samples after rewinding, got %d", len(first.Data), len(second.Data))
	}
	for i := range first.Data {
		if first.Data[i] != second.Data[i] {
			t.Fatalf("sample %d differs after rewinding: %d != %d", i, first.Data[i], second.Data[i])
		}
	}
}

// TestMSADPCMFmtErrors checks that a malformed fmt extension is rejected
// instead of producing garbage or silence.
func TestMSADPCMFmtErrors(t *testing.T) {
	block := msADPCMBlock(0, 16, 0, 0, make([]byte, 25))
	tests := []struct {
		name    string
		numChan int
		align   int
		spb     int
		coefs   [][2]int16
	}{
		{name: "samples per block exceeds the block capacity", numChan: 1, align: 32, spb: 100, coefs: msADPCMStandardCoefs},
		{name: "block alignment too small", numChan: 1, align: 6, spb: 1, coefs: msADPCMStandardCoefs},
		{name: "no coefficients", numChan: 1, align: 32, spb: 52, coefs: nil},
		{name: "more than two channels", numChan: 3, align: 32, spb: 52, coefs: msADPCMStandardCoefs},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			wavData := msADPCMWav(tc.numChan, tc.align, tc.spb, tc.coefs, -1, block)
			d := newMSADPCMDecoder(t, wavData)
			if _, err := d.FullPCMBuffer(); err == nil {
				t.Fatal("expected an error for a malformed fmt extension")
			}
		})
	}
}

// TestMSADPCMReferenceFiles decodes fixtures and compares the output to an
// independent decoder. The mono and stereo fixtures are hand built blocks;
// the kick fixture was encoded with FFmpeg and decoded with a second,
// independent implementation, so this also covers the fact chunk trimming.
func TestMSADPCMReferenceFiles(t *testing.T) {
	monoWant := []int{0, 1000, 984, 952, 952, 984, 936, 888, 936, 952, 904, 888, 904, 856,
		872, 840, 792, 744, 744, 744, 696, 664, 616, 632, 632, 584, 632, 648, 600, 568,
		600, 632, 648, 600, 616, 632, 632, 584, 552, 504, 520, 568, 536, 520, 520, 488,
		504, 456, 472, 456, 472, 520}
	stereoWant := []int{0, 500, 1000, -2000, 936, -4200, 860, -5743, 992, -6432, 1212,
		-7681, 1702, -7315, 2871, -7305, 4471, -5738, 5429, -6103, 1989, -6304, -3171,
		-4508, -79, -4649, -1467, -7690, 6015, -4040, -1467, 12989, 11973, 17030,
		-10427, 15739, -32768, 32767, 18724, 15130}

	tests := []struct {
		path    string
		chans   int
		samples []int
	}{
		{"fixtures/adpcm-ms-mono.wav", 1, monoWant},
		{"fixtures/adpcm-ms-stereo.wav", 2, stereoWant},
	}
	for _, tc := range tests {
		t.Run(tc.path, func(t *testing.T) {
			f, err := os.Open(tc.path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			buf, err := NewDecoder(f).FullPCMBuffer()
			if err != nil {
				t.Fatal(err)
			}
			if buf.Format.NumChannels != tc.chans {
				t.Fatalf("expected %d channels, got %d", tc.chans, buf.Format.NumChannels)
			}
			if !reflect.DeepEqual(buf.Data, tc.samples) {
				t.Fatalf("decoded samples differ from the reference:\n got %v\nwant %v", buf.Data, tc.samples)
			}
		})
	}

	t.Run("fixtures/adpcm-ms-kick.wav", func(t *testing.T) {
		f, err := os.Open("fixtures/adpcm-ms-kick.wav")
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		d := NewDecoder(f)
		buf, err := d.FullPCMBuffer()
		if err != nil {
			t.Fatal(err)
		}
		// The fact chunk announces 4484 frames, which is less than the
		// 6108 samples the packed blocks hold: the tail must be dropped.
		if len(buf.Data) != 4484 {
			t.Fatalf("expected the fact chunk to trim the output to 4484 samples, got %d", len(buf.Data))
		}
		want := []int{76, 75, 75, 75, 75, 75, 75, 75, 75, 59, 75, 75, 187, 453, 1090, 2616}
		for i, w := range want {
			if buf.Data[i] != w {
				t.Fatalf("sample %d: expected %d, got %d", i, w, buf.Data[i])
			}
		}
		if got := buf.Data[len(buf.Data)-1]; got != 153 {
			t.Fatalf("unexpected last sample: %d", got)
		}
	})
}

// slowReader hands out at most step bytes per Read call, so blocks are split
// across multiple reads.
type slowReader struct {
	r    io.Reader
	step int
}

func (s *slowReader) Read(p []byte) (int, error) {
	if len(p) > s.step {
		p = p[:s.step]
	}
	return s.r.Read(p)
}

// TestMSADPCMPartialReads checks that a block spanning several reads is
// reassembled instead of being decoded as truncated.
func TestMSADPCMPartialReads(t *testing.T) {
	block := msADPCMBlock(0, 16, 1000, 0, make([]byte, 25))
	wavData := msADPCMWav(1, 32, 52, msADPCMStandardCoefs, 52, block)

	want, err := newMSADPCMDecoder(t, wavData).FullPCMBuffer()
	if err != nil {
		t.Fatal(err)
	}

	// Write the container to disk, then only allow small reads.
	f, err := os.CreateTemp(t.TempDir(), "adpcm-slow-*.wav")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	if _, err := f.Write(wavData); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	d := NewDecoder(&seekableSlowReader{f: f, step: 5})
	got, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Data, want.Data) {
		t.Fatalf("partial reads changed the output:\n got %v\nwant %v", got.Data, want.Data)
	}
}

type seekableSlowReader struct {
	f    *os.File
	step int
}

func (s *seekableSlowReader) Read(p []byte) (int, error) {
	if len(p) > s.step {
		p = p[:s.step]
	}
	return s.f.Read(p)
}

func (s *seekableSlowReader) Seek(offset int64, whence int) (int64, error) {
	return s.f.Seek(offset, whence)
}

// TestMSADPCMHeaderErrorIsSticky makes sure a malformed fmt extension is
// reported even when the caller reads the headers first and then asks for
// samples, instead of falling through to the raw PCM decoding path.
func TestMSADPCMHeaderErrorIsSticky(t *testing.T) {
	block := msADPCMBlock(0, 16, 0, 0, make([]byte, 25))
	// 32 byte block holds at most 52 frames: 100 is impossible.
	wavData := msADPCMWav(1, 32, 100, msADPCMStandardCoefs, -1, block)
	d := newMSADPCMDecoder(t, wavData)
	d.ReadInfo()
	if d.Err() == nil {
		t.Fatal("expected ReadInfo to report the malformed fmt extension")
	}
	buf := &audio.IntBuffer{Data: make([]int, 10)}
	if _, err := d.PCMBuffer(buf); err == nil {
		t.Fatal("expected PCMBuffer to report the malformed fmt extension")
	}
	if _, err := d.FullPCMBuffer(); err == nil {
		t.Fatal("expected FullPCMBuffer to report the malformed fmt extension")
	}
}

// TestMSADPCMOddDataSize checks that the RIFF padding byte that follows an odd
// sized data chunk isn't decoded as an extra differential code.
func TestMSADPCMOddDataSize(t *testing.T) {
	// block alignment 31 => header 7 + 24 body bytes, 50 frames per block
	block := msADPCMBlock(0, 16, 1000, 0, make([]byte, 24))
	wavData := msADPCMWav(1, 31, 50, msADPCMStandardCoefs, 50, block)
	d := newMSADPCMDecoder(t, wavData)
	buf, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatal(err)
	}
	if len(buf.Data) != 50 {
		t.Fatalf("expected 50 samples, got %d", len(buf.Data))
	}
	if buf.Data[len(buf.Data)-1] != 1000 {
		t.Fatalf("unexpected last sample: %d", buf.Data[len(buf.Data)-1])
	}
}

// TestMSADPCMFactBeforeDataIsHonored checks that a fact chunk placed before the
// data chunk trims the decoded output.
func TestMSADPCMFactBeforeDataIsHonored(t *testing.T) {
	block := msADPCMBlock(0, 16, 1000, 0, make([]byte, 25))
	wavData := msADPCMWav(1, 32, 52, msADPCMStandardCoefs, 10, block)

	d := newMSADPCMDecoder(t, wavData)
	buf, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatal(err)
	}
	if len(buf.Data) != 10 {
		t.Fatalf("expected the fact chunk to trim the output to 10 samples, got %d", len(buf.Data))
	}
}

// TestMSADPCMDeclaredSamplesPerBlock checks that a declared samples per block
// lower than the block capacity is honored: the trailing codes are ignored.
func TestMSADPCMDeclaredSamplesPerBlock(t *testing.T) {
	// 32 byte block can hold 52 frames; declare 10.
	block := msADPCMBlock(0, 16, 1000, 0, make([]byte, 25))
	wavData := msADPCMWav(1, 32, 10, msADPCMStandardCoefs, -1, block)
	d := newMSADPCMDecoder(t, wavData)
	buf, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatal(err)
	}
	if len(buf.Data) != 10 {
		t.Fatalf("expected the declared 10 samples per block, got %d", len(buf.Data))
	}
	// Streaming must respect the same limit.
	d = newMSADPCMDecoder(t, wavData)
	buf = &audio.IntBuffer{Data: make([]int, 10)}
	n, err := d.PCMBuffer(buf)
	if err != nil {
		t.Fatal(err)
	}
	if n != 10 {
		t.Fatalf("expected 10 streamed samples, got %d", n)
	}
	if n, _ = d.PCMBuffer(buf); n != 0 {
		t.Fatalf("expected the stream to be exhausted, got %d more samples", n)
	}
}

// TestMSADPCMWrongBitDepth checks that an MS ADPCM stream that doesn't declare
// 4 bits per sample is rejected.
func TestMSADPCMWrongBitDepth(t *testing.T) {
	block := msADPCMBlock(0, 16, 0, 0, make([]byte, 25))
	wavData := msADPCMWav(1, 32, 52, msADPCMStandardCoefs, -1, block)
	// The fmt chunk layout from msADPCMWav puts bits per sample at offset
	// 12+8+14 inside the file.
	binary.LittleEndian.PutUint16(wavData[12+8+14:], 8)
	d := newMSADPCMDecoder(t, wavData)
	if _, err := d.FullPCMBuffer(); err == nil {
		t.Fatal("expected an error for a non 4 bit MS ADPCM stream")
	}
}

// TestMSADPCMFactChunkErrors checks that a malformed fact chunk is rejected
// for Microsoft ADPCM instead of being ignored.
func TestMSADPCMFactChunkErrors(t *testing.T) {
	block := msADPCMBlock(0, 16, 1000, 0, make([]byte, 25))
	base := msADPCMWav(1, 32, 52, msADPCMStandardCoefs, -1, block)

	// Insert a 2 byte fact chunk (not enough for the uint32 frame count)
	// right after the fmt chunk.
	insertAt := 12 + 8 + 50 // RIFF header + fmt id/size + fmt payload
	var fact bytes.Buffer
	fact.Write([]byte("fact"))
	binary.Write(&fact, binary.LittleEndian, uint32(2))
	fact.Write([]byte{0, 0})
	wavData := append(append(append([]byte{}, base[:insertAt]...), fact.Bytes()...), base[insertAt:]...)
	binary.LittleEndian.PutUint32(wavData[4:8], uint32(len(wavData)-8))

	d := newMSADPCMDecoder(t, wavData)
	if _, err := d.FullPCMBuffer(); err == nil {
		t.Fatal("expected an error for a malformed fact chunk")
	}
}

func TestMSADPCMDoesNotAffectPCM(t *testing.T) {
	// Sanity check: the PCM fixtures keep decoding identically after the fmt
	// chunk parsing change.
	for _, path := range []string{"fixtures/kick.wav", "fixtures/bass.wav"} {
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		d := NewDecoder(f)
		buf, err := d.FullPCMBuffer()
		f.Close()
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		if d.WavAudioFormat != 1 {
			t.Fatalf("%s: expected PCM format 1, got %d", path, d.WavAudioFormat)
		}
		if len(buf.Data) == 0 {
			t.Fatalf("%s: no samples decoded", path)
		}
	}
}

// TestMSADPCMSamplesPerBlock guards the samples per block formula.
func TestMSADPCMSamplesPerBlock(t *testing.T) {
	tests := []struct {
		blockAlign int
		channels   int
		want       int
	}{
		{256, 1, 500},
		{256, 2, 244},
		{1024, 1, 2036},
		{1024, 2, 1012},
		{32, 1, 52},
		{32, 2, 20},
	}
	for _, tc := range tests {
		if got := msADPCMSamplesPerBlock(tc.blockAlign, tc.channels); got != tc.want {
			t.Errorf("msADPCMSamplesPerBlock(%d, %d) = %d, want %d",
				tc.blockAlign, tc.channels, got, tc.want)
		}
	}
}
