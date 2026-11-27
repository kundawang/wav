package wav

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"testing"

	"github.com/go-audio/audio"
)

// The GUIDs below are declared independently from the production constants so
// the handcrafted byte tests don't simply compare the package against itself.
// They are the Sony Wave64 well-known GUIDs.
var (
	testRiffGUID = [16]byte{
		0x72, 0x69, 0x66, 0x66, 0x2E, 0x91, 0xCF, 0x11,
		0xA5, 0xD6, 0x28, 0xDB, 0x04, 0xC1, 0x00, 0x00,
	}
	testWaveGUID = [16]byte{
		0x77, 0x61, 0x76, 0x65, 0xF3, 0xAC, 0xD3, 0x11,
		0x8C, 0xD1, 0x00, 0xC0, 0x4F, 0x8E, 0xDB, 0x8A,
	}
	testFmtGUID = [16]byte{
		0x66, 0x6D, 0x74, 0x20, 0xF3, 0xAC, 0xD3, 0x11,
		0x8C, 0xD1, 0x00, 0xC0, 0x4F, 0x8E, 0xDB, 0x8A,
	}
	testDataGUID = [16]byte{
		0x64, 0x61, 0x74, 0x61, 0xF3, 0xAC, 0xD3, 0x11,
		0x8C, 0xD1, 0x00, 0xC0, 0x4F, 0x8E, 0xDB, 0x8A,
	}
)

// memWriteSeeker is an in-memory io.WriteSeeker supporting overwrite after
// Seek, which the encoder needs to backfill size fields.
type memWriteSeeker struct {
	data []byte
	pos  int64
}

func (m *memWriteSeeker) Write(p []byte) (int, error) {
	end := m.pos + int64(len(p))
	if int64(len(m.data)) < end {
		m.data = append(m.data, make([]byte, end-int64(len(m.data)))...)
	}
	copy(m.data[m.pos:end], p)
	m.pos = end
	return len(p), nil
}

func (m *memWriteSeeker) Seek(offset int64, whence int) (int64, error) {
	var abs int64
	switch whence {
	case io.SeekStart:
		abs = offset
	case io.SeekCurrent:
		abs = m.pos + offset
	case io.SeekEnd:
		abs = int64(len(m.data)) + offset
	}
	if abs < 0 {
		return 0, errors.New("negative position")
	}
	m.pos = abs
	return abs, nil
}

// w64File is the result of the independent Wave64 decoder below.
type w64File struct {
	audioFormat    uint16
	numChannels    uint16
	sampleRate     uint32
	avgBytesPerSec uint32
	blockAlign     uint16
	bitsPerSample  uint16

	riffSize uint64
	dataSize uint64

	chunkOffsets []int64
	// payload is the data chunk content including any alignment padding.
	payload []byte
}

// parseWave64 is a deliberately standalone Wave64 reader. It only uses the
// standard library and the test-local GUIDs above, never the encoder's
// internals, so a round trip through it is an independent verification.
func parseWave64(t *testing.T, raw []byte) w64File {
	t.Helper()
	r := bytes.NewReader(raw)
	out := w64File{}

	readGUID := func() [16]byte {
		var g [16]byte
		if _, err := io.ReadFull(r, g[:]); err != nil {
			t.Fatalf("reading GUID: %v", err)
		}
		return g
	}
	readU64 := func() uint64 {
		var b [8]byte
		if _, err := io.ReadFull(r, b[:]); err != nil {
			t.Fatalf("reading uint64: %v", err)
		}
		return binary.LittleEndian.Uint64(b[:])
	}

	if g := readGUID(); g != testRiffGUID {
		t.Fatalf("RIFF GUID mismatch: %x", g)
	}
	out.riffSize = readU64()
	if out.riffSize != uint64(len(raw)) {
		t.Fatalf("RIFF size = %d, want total file length %d (chunk size must include its header)", out.riffSize, len(raw))
	}
	if g := readGUID(); g != testWaveGUID {
		t.Fatalf("WAVE GUID mismatch: %x", g)
	}

	sawFmt, sawData := false, false
	for {
		off, err := r.Seek(0, io.SeekCurrent)
		if err != nil {
			t.Fatal(err)
		}
		if off%8 != 0 {
			t.Fatalf("chunk at offset %d is not 8 byte aligned", off)
		}

		g, err := func() ([16]byte, error) {
			var g [16]byte
			_, e := io.ReadFull(r, g[:])
			return g, e
		}()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("reading chunk GUID: %v", err)
		}
		size := readU64()
		if size < 24 {
			t.Fatalf("chunk size %d smaller than its 24 byte header", size)
		}
		out.chunkOffsets = append(out.chunkOffsets, off)
		payloadLen := int64(size - 24)

		switch g {
		case testFmtGUID:
			if sawFmt {
				t.Fatal("duplicate fmt chunk")
			}
			sawFmt = true
			if payloadLen != 16 {
				t.Fatalf("fmt payload = %d bytes, want 16", payloadLen)
			}
			body := make([]byte, 16)
			if _, err := io.ReadFull(r, body); err != nil {
				t.Fatal(err)
			}
			out.audioFormat = binary.LittleEndian.Uint16(body[0:2])
			out.numChannels = binary.LittleEndian.Uint16(body[2:4])
			out.sampleRate = binary.LittleEndian.Uint32(body[4:8])
			out.avgBytesPerSec = binary.LittleEndian.Uint32(body[8:12])
			out.blockAlign = binary.LittleEndian.Uint16(body[12:14])
			out.bitsPerSample = binary.LittleEndian.Uint16(body[14:16])
		case testDataGUID:
			if sawData {
				t.Fatal("duplicate data chunk")
			}
			sawData = true
			out.dataSize = size
			out.payload = make([]byte, payloadLen)
			if _, err := io.ReadFull(r, out.payload); err != nil {
				t.Fatal(err)
			}
		default:
			t.Fatalf("unexpected GUID %x at offset %d", g, off)
		}
	}

	if !sawFmt || !sawData {
		t.Fatalf("missing chunks: fmt=%v data=%v", sawFmt, sawData)
	}
	return out
}

// decodePCMSamples decodes little-endian signed PCM samples independently
// from the production decoder.
func decodePCMSamples(t *testing.T, b []byte, bits uint16) []int {
	t.Helper()
	bytesPerSample := int(bits / 8)
	if len(b)%bytesPerSample != 0 {
		t.Fatalf("payload length %d not a multiple of %d bytes", len(b), bytesPerSample)
	}
	out := make([]int, 0, len(b)/bytesPerSample)
	for i := 0; i < len(b); i += bytesPerSample {
		switch bits {
		case 16:
			out = append(out, int(int16(binary.LittleEndian.Uint16(b[i:i+2]))))
		case 24:
			v := int32(b[i]) | int32(b[i+1])<<8 | int32(b[i+2])<<16
			// sign extend 24 bit
			if v&0x800000 != 0 {
				v |= -1 << 24
			}
			out = append(out, int(v))
		default:
			t.Fatalf("unsupported bit depth %d", bits)
		}
	}
	return out
}

func intBuffer(sampleRate, channels, bits int, data []int) *audio.IntBuffer {
	return &audio.IntBuffer{
		Format:         &audio.Format{SampleRate: sampleRate, NumChannels: channels},
		Data:           data,
		SourceBitDepth: bits,
	}
}

func TestNewWave64Encoder_UnsupportedParams(t *testing.T) {
	ws := &memWriteSeeker{}
	cases := []struct {
		name                       string
		sampleRate, bits, channels int
	}{
		{"8 bit not supported", 48000, 8, 1},
		{"32 bit not supported", 48000, 32, 1},
		{"3 channels not supported", 48000, 16, 3},
		{"zero channels not supported", 48000, 16, 0},
		{"zero sample rate not supported", 0, 16, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e, err := NewWave64Encoder(ws, tc.sampleRate, tc.bits, tc.channels)
			if err == nil {
				t.Fatalf("expected an error for %s", tc.name)
			}
			if e != nil {
				t.Fatal("expected nil encoder on error")
			}
		})
	}

	e, err := NewWave64Encoder(ws, 48000, 24, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !e.IsWave64() {
		t.Error("IsWave64() = false, want true")
	}
	if e.WavAudioFormat != 1 {
		t.Errorf("WavAudioFormat = %d, want 1 (PCM)", e.WavAudioFormat)
	}
	if NewEncoder(ws, 48000, 16, 1, 1).IsWave64() {
		t.Error("classic NewEncoder must not report Wave64")
	}
}

func TestWave64Encoder_HandcraftedBytes(t *testing.T) {
	// Two 16 bit mono frames: 0x0102 and -2 (0xFFFE).
	samples := []int{0x0102, -2}
	ws := &memWriteSeeker{}
	e, err := NewWave64Encoder(ws, 48000, 16, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Write(intBuffer(48000, 1, 16, samples)); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}

	// Build the exact expected byte sequence by hand.
	var want bytes.Buffer
	put := func(v interface{}) {
		if err := binary.Write(&want, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	want.Write(testRiffGUID[:])
	put(uint64(112)) // total file length, RIFF size includes the 24 byte header
	want.Write(testWaveGUID[:])
	want.Write(testFmtGUID[:])
	put(uint64(40)) // fmt chunk: 24 header + 16 payload
	put(uint16(1))  // PCM
	put(uint16(1))  // mono
	put(uint32(48000))
	put(uint32(96000)) // avg bytes/sec
	put(uint16(2))     // block align
	put(uint16(16))    // bits per sample
	want.Write(testDataGUID[:])
	// 24 header + 4 payload + 4 zero padding bytes so the chunk ends on an
	// 8 byte boundary (payload ends at file offset 108, padding reaches 112).
	put(uint64(32))
	put(int16(0x0102))
	put(int16(-2))
	want.Write([]byte{0, 0, 0, 0})

	if !bytes.Equal(ws.data, want.Bytes()) {
		t.Fatalf("byte mismatch\n got: %x\nwant: %x", ws.data, want.Bytes())
	}
	if len(ws.data) != 112 {
		t.Fatalf("file length = %d, want 112", len(ws.data))
	}

	parsed := parseWave64(t, ws.data)
	if parsed.riffSize != 112 || parsed.dataSize != 32 {
		t.Fatalf("sizes = riff %d data %d, want 112/32", parsed.riffSize, parsed.dataSize)
	}
	if parsed.chunkOffsets[0] != 40 || parsed.chunkOffsets[1] != 80 {
		t.Fatalf("chunk offsets = %v, want [40 80]", parsed.chunkOffsets)
	}
	if len(parsed.payload) != 8 {
		t.Fatalf("data payload = %d bytes, want 8 (4 audio + 4 padding)", len(parsed.payload))
	}
	for i := 4; i < 8; i++ {
		if parsed.payload[i] != 0 {
			t.Fatalf("padding byte at %d = %#x, want 0", i, parsed.payload[i])
		}
	}
	got := decodePCMSamples(t, parsed.payload[:4], 16)
	if len(got) != 2 || got[0] != 0x0102 || got[1] != -2 {
		t.Fatalf("samples = %v, want [258 -2]", got)
	}
}

func TestWave64Encoder_RoundTrip(t *testing.T) {
	cases := []struct {
		name           string
		bits, channels int
		samples        []int
		wantDataSize   uint64
		wantPadding    int
		wantFileLength int
		wantAudioBytes int
	}{
		{
			name: "16bit mono, 4 byte payload, 4 byte padding",
			bits: 16, channels: 1,
			samples:        []int{123, -456},
			wantDataSize:   32,
			wantPadding:    4,
			wantFileLength: 112,
			wantAudioBytes: 4,
		},
		{
			name: "24bit mono, 6 byte payload, 2 byte padding",
			bits: 24, channels: 1,
			samples:        []int{0x7FFFFF, -0x800000},
			wantDataSize:   32,
			wantPadding:    2,
			wantFileLength: 112,
			wantAudioBytes: 6,
		},
		{
			name: "16bit stereo, 8 byte payload, no padding",
			bits: 16, channels: 2,
			samples:        []int{1, 2, 3, -4},
			wantDataSize:   32,
			wantPadding:    0,
			wantFileLength: 112,
			wantAudioBytes: 8,
		},
		{
			name: "24bit stereo, 12 byte payload, 4 byte padding",
			bits: 24, channels: 2,
			samples:        []int{10, 20, -30, 40},
			wantDataSize:   40,
			wantPadding:    4,
			wantFileLength: 120,
			wantAudioBytes: 12,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ws := &memWriteSeeker{}
			e, err := NewWave64Encoder(ws, 44100, tc.bits, tc.channels)
			if err != nil {
				t.Fatal(err)
			}

			// Split the samples across multiple Write calls at frame
			// boundaries so each buffer holds an integer number of frames.
			frameSize := tc.channels
			totalFrames := len(tc.samples) / frameSize
			firstFrames := totalFrames / 2
			writes := [][]int{
				tc.samples[:firstFrames*frameSize],
				tc.samples[firstFrames*frameSize : (firstFrames+1)*frameSize],
				tc.samples[(firstFrames+1)*frameSize:],
			}
			for _, part := range writes {
				if len(part) == 0 {
					continue
				}
				if err := e.Write(intBuffer(44100, tc.channels, tc.bits, part)); err != nil {
					t.Fatal(err)
				}
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}

			if len(ws.data) != tc.wantFileLength {
				t.Fatalf("file length = %d, want %d", len(ws.data), tc.wantFileLength)
			}
			if len(ws.data)%8 != 0 {
				t.Fatalf("file length %d not 8 byte aligned", len(ws.data))
			}

			parsed := parseWave64(t, ws.data)
			if parsed.dataSize != tc.wantDataSize {
				t.Errorf("data chunk size = %d, want %d", parsed.dataSize, tc.wantDataSize)
			}
			if parsed.riffSize != uint64(tc.wantFileLength) {
				t.Errorf("riff size = %d, want %d", parsed.riffSize, tc.wantFileLength)
			}
			if parsed.audioFormat != 1 || parsed.bitsPerSample != uint16(tc.bits) ||
				parsed.numChannels != uint16(tc.channels) || parsed.sampleRate != 44100 {
				t.Fatalf("fmt body = format %d, %d bits, %d ch, %d Hz",
					parsed.audioFormat, parsed.bitsPerSample, parsed.numChannels, parsed.sampleRate)
			}
			if int(parsed.blockAlign) != tc.channels*tc.bits/8 {
				t.Errorf("block align = %d", parsed.blockAlign)
			}
			if int(parsed.avgBytesPerSec) != 44100*tc.channels*tc.bits/8 {
				t.Errorf("avg bytes/sec = %d", parsed.avgBytesPerSec)
			}

			if got := len(parsed.payload) - tc.wantAudioBytes; got != tc.wantPadding {
				t.Fatalf("padding = %d bytes, want %d", got, tc.wantPadding)
			}
			for i := tc.wantAudioBytes; i < len(parsed.payload); i++ {
				if parsed.payload[i] != 0 {
					t.Fatalf("padding byte at %d = %#x, want 0", i, parsed.payload[i])
				}
			}

			got := decodePCMSamples(t, parsed.payload[:tc.wantAudioBytes], uint16(tc.bits))
			if len(got) != len(tc.samples) {
				t.Fatalf("decoded %d samples, want %d", len(got), len(tc.samples))
			}
			for i, v := range tc.samples {
				// 24 bit samples are range limited, 16 bit values fit as-is.
				want := v
				if tc.bits == 24 {
					want = v << 8 >> 8
				}
				if got[i] != want {
					t.Fatalf("sample %d = %d, want %d", i, got[i], want)
				}
			}
		})
	}
}

func TestWave64Encoder_MultipleWritesMatchSingleWrite(t *testing.T) {
	samples := []int{100, -200, 300, -400, 500, -600}

	single := &memWriteSeeker{}
	e1, err := NewWave64Encoder(single, 48000, 16, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := e1.Write(intBuffer(48000, 2, 16, samples)); err != nil {
		t.Fatal(err)
	}
	if err := e1.Close(); err != nil {
		t.Fatal(err)
	}

	multi := &memWriteSeeker{}
	e2, err := NewWave64Encoder(multi, 48000, 16, 2)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len(samples); i += 2 {
		if err := e2.Write(intBuffer(48000, 2, 16, samples[i:i+2])); err != nil {
			t.Fatal(err)
		}
	}
	if err := e2.Close(); err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(single.data, multi.data) {
		t.Fatalf("chunked writes produced different bytes\n single: %x\n  multi: %x", single.data, multi.data)
	}
}

func TestWave64Encoder_EmptyAudio(t *testing.T) {
	ws := &memWriteSeeker{}
	e, err := NewWave64Encoder(ws, 48000, 16, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Close without any Write: the file must still be a valid Wave64 with an
	// empty data chunk.
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if len(ws.data) != 104 { // 24 RIFF + 16 WAVE + 40 fmt + 24 empty data
		t.Fatalf("empty file length = %d, want 104", len(ws.data))
	}
	parsed := parseWave64(t, ws.data)
	if parsed.dataSize != 24 {
		t.Errorf("empty data chunk size = %d, want 24 (header only)", parsed.dataSize)
	}
	if len(parsed.payload) != 0 {
		t.Errorf("payload length = %d, want 0", len(parsed.payload))
	}
	if len(decodePCMSamples(t, parsed.payload, 16)) != 0 {
		t.Error("expected no samples")
	}

	// An explicit Write of an empty buffer must not fail either.
	ws2 := &memWriteSeeker{}
	e2, err := NewWave64Encoder(ws2, 48000, 24, 2)
	if err != nil {
		t.Fatal(err)
	}
	empty := intBuffer(48000, 2, 24, []int{})
	if err := e2.Write(empty); err != nil {
		t.Fatal(err)
	}
	if err := e2.Close(); err != nil {
		t.Fatal(err)
	}
	parseWave64(t, ws2.data)
}

func TestWave64Encoder_NilBuffer(t *testing.T) {
	ws := &memWriteSeeker{}
	e, err := NewWave64Encoder(ws, 48000, 16, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Write(nil); err == nil {
		t.Fatal("expected an error when writing a nil buffer")
	}
}

func TestWave64Encoder_MetadataRejected(t *testing.T) {
	ws := &memWriteSeeker{}
	e, err := NewWave64Encoder(ws, 48000, 16, 1)
	if err != nil {
		t.Fatal(err)
	}
	e.Metadata = &Metadata{Artist: "someone"}
	if err := e.Write(intBuffer(48000, 1, 16, []int{1, 2})); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); !errors.Is(err, ErrWave64Metadata) {
		t.Fatalf("expected ErrWave64Metadata, got %v", err)
	}
}

func TestWave64Encoder_DoubleCloseIsIdempotent(t *testing.T) {
	ws := &memWriteSeeker{}
	e, err := NewWave64Encoder(ws, 48000, 16, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Write(intBuffer(48000, 1, 16, []int{7, -7})); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	first := append([]byte(nil), ws.data...)
	if err := e.Close(); err != nil {
		t.Fatalf("second Close returned error: %v", err)
	}
	if !bytes.Equal(first, ws.data) {
		t.Fatal("second Close mutated the file (duplicated padding or size fields)")
	}
	parsed := parseWave64(t, ws.data)
	if parsed.riffSize != uint64(len(ws.data)) {
		t.Fatalf("riff size = %d, file length = %d", parsed.riffSize, len(ws.data))
	}

	// The classic RIFF encoder must be idempotent as well.
	ws2 := &memWriteSeeker{}
	e2 := NewEncoder(ws2, 48000, 16, 1, 1)
	if err := e2.Write(intBuffer(48000, 1, 16, []int{1})); err != nil {
		t.Fatal(err)
	}
	if err := e2.Close(); err != nil {
		t.Fatal(err)
	}
	first2 := append([]byte(nil), ws2.data...)
	if err := e2.Close(); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first2, ws2.data) {
		t.Fatal("second Close mutated the RIFF file")
	}
}

func TestWave64Encoder_WriteFrame(t *testing.T) {
	ws := &memWriteSeeker{}
	e, err := NewWave64Encoder(ws, 48000, 16, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.WriteFrame(int16(0x1234)); err != nil {
		t.Fatal(err)
	}
	if err := e.WriteFrame(int16(-2)); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	parsed := parseWave64(t, ws.data)
	if got := decodePCMSamples(t, parsed.payload[:4], 16); len(got) != 2 || got[0] != 0x1234 || got[1] != -2 {
		t.Fatalf("WriteFrame samples = %v, want [4660 -2]", got)
	}
}

func TestWave64DataSize(t *testing.T) {
	cases := []struct {
		frames, channels, bits int
		want                   uint64
	}{
		{0, 1, 16, 24}, // header only
		{2, 1, 16, 32}, // 4 payload + 4 padding
		{2, 1, 24, 32}, // 6 payload + 2 padding
		{2, 2, 24, 40}, // 12 payload + 4 padding
		{3, 2, 16, 40}, // 12 payload + 4 padding
		{4, 2, 16, 40}, // 16 payload, no padding
	}
	for _, tc := range cases {
		got, err := wave64DataSize(tc.frames, tc.channels, tc.bits)
		if err != nil {
			t.Fatalf("wave64DataSize(%v): %v", tc, err)
		}
		if got != tc.want {
			t.Errorf("wave64DataSize(%d, %d, %d) = %d, want %d",
				tc.frames, tc.channels, tc.bits, got, tc.want)
		}
		if got%8 != 0 {
			t.Errorf("size %d not 8 byte aligned", got)
		}
	}

	if _, err := wave64DataSize(-1, 1, 16); err == nil {
		t.Error("expected error for negative frames")
	}

	// 64 bit overflow: MaxInt frames of stereo 24 bit cannot fit a uint64.
	if _, err := wave64DataSize(math.MaxInt, 2, 24); !errors.Is(err, ErrWave64Overflow) {
		t.Fatalf("expected ErrWave64Overflow, got %v", err)
	}
	if _, err := wave64DataSize(math.MaxInt64, 1, 16); !errors.Is(err, ErrWave64Overflow) {
		t.Fatalf("expected ErrWave64Overflow, got %v", err)
	}
}

func TestRIFFEncoder_KeepsClassicFormat(t *testing.T) {
	// The classic encoder must still emit the 32 bit RIFF layout.
	ws := &memWriteSeeker{}
	e := NewEncoder(ws, 48000, 16, 1, 1)
	if err := e.Write(intBuffer(48000, 1, 16, []int{1, 2})); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if string(ws.data[:4]) != "RIFF" {
		t.Fatalf("classic output starts with %q, want RIFF", ws.data[:4])
	}
	if string(ws.data[8:12]) != "WAVE" {
		t.Fatalf("missing WAVE magic at offset 8: %x", ws.data[8:12])
	}
	if string(ws.data[12:16]) != "fmt " {
		t.Fatalf("missing fmt FourCC: %x", ws.data[12:16])
	}
	if uint32(binary.LittleEndian.Uint32(ws.data[4:8])) != uint32(len(ws.data)-8) {
		t.Error("RIFF size field not backfilled as file length minus 8")
	}

	// Wave64 output must not start with the RIFF FourCC.
	w64 := &memWriteSeeker{}
	e64, err := NewWave64Encoder(w64, 48000, 16, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := e64.Write(intBuffer(48000, 1, 16, []int{1, 2})); err != nil {
		t.Fatal(err)
	}
	if err := e64.Close(); err != nil {
		t.Fatal(err)
	}
	if string(w64.data[:4]) != "riff" {
		t.Fatalf("Wave64 output starts with %q, want riff GUID", w64.data[:4])
	}
}

func TestRIFFEncoder_OverflowDetected(t *testing.T) {
	ws := &memWriteSeeker{}
	e := NewEncoder(ws, 48000, 16, 2, 1)
	if err := e.Write(intBuffer(48000, 2, 16, []int{1, 2})); err != nil {
		t.Fatal(err)
	}
	// Simulate a long recording whose data chunk exceeds uint32.
	e.frames = math.MaxInt32
	err := e.Close()
	if !errors.Is(err, ErrWave64Overflow) {
		t.Fatalf("expected ErrWave64Overflow, got %v", err)
	}
}
