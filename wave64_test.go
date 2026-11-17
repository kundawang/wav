package wav

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/go-audio/audio"
)

// w64GUID builds a WAVE64 chunk GUID from a 4-byte RIFF-style tag using
// the standard {xxxxxxxx-ACF3-11D3-8CD1-00C04F8EDB8A} pattern.
func w64GUID(tag string) [16]byte {
	var g [16]byte
	copy(g[:4], tag)
	copy(g[4:], wave64ChunkGUIDTail)
	return g
}

type w64TestChunk struct {
	guid [16]byte
	data []byte
	// sizeOverride, when > 0, replaces the computed chunk size
	// (used to build malformed fixtures).
	sizeOverride uint64
}

// buildWave64 assembles a WAVE64 container from the passed chunks.
func buildWave64(chunks ...w64TestChunk) []byte {
	buf := &bytes.Buffer{}
	buf.Write(wave64RiffGUID[:])
	buf.Write(make([]byte, 8)) // placeholder for the total size
	buf.Write(wave64WaveGUID[:])
	for _, c := range chunks {
		size := uint64(wave64ChunkHeaderSize + len(c.data))
		if c.sizeOverride > 0 {
			size = c.sizeOverride
		}
		buf.Write(c.guid[:])
		binary.Write(buf, binary.LittleEndian, size)
		buf.Write(c.data)
		// chunk data is zero padded to an 8-byte boundary
		if pad := pad8(int(size)) - int(size); pad > 0 {
			buf.Write(make([]byte, pad))
		}
	}
	out := buf.Bytes()
	binary.LittleEndian.PutUint64(out[16:24], uint64(len(out)))
	return out
}

func pad8(n int) int {
	return (n + 7) &^ 7
}

// w64Fmt returns the content of a fmt chunk.
func w64Fmt(format, numChans uint16, sampleRate uint32, bitDepth uint16) []byte {
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint16(buf[0:2], format)
	binary.LittleEndian.PutUint16(buf[2:4], numChans)
	binary.LittleEndian.PutUint32(buf[4:8], sampleRate)
	blockAlign := numChans * (bitDepth / 8)
	binary.LittleEndian.PutUint32(buf[8:12], sampleRate*uint32(blockAlign))
	binary.LittleEndian.PutUint16(buf[12:14], blockAlign)
	binary.LittleEndian.PutUint16(buf[14:16], bitDepth)
	return buf
}

// pcm16Fixture is a 16-bit mono PCM WAVE64 file holding an unknown chunk
// with an odd data size (exercising the 8-byte alignment) and a data chunk
// whose size is not a multiple of 8.
func pcm16Fixture() ([]byte, []int) {
	samples := []int{0, 1000, -1000, 32767, -32768}
	data := &bytes.Buffer{}
	for _, s := range samples {
		binary.Write(data, binary.LittleEndian, int16(s))
	}
	return buildWave64(
		w64TestChunk{guid: [16]byte{0x01, 0x02, 0x03, 0x04}, data: []byte{1, 2, 3}}, // unknown, odd size
		w64TestChunk{guid: w64GUID("fmt "), data: w64Fmt(1, 1, 44100, 16)},
		w64TestChunk{guid: [16]byte{0xAA, 0xBB}, data: []byte{9, 8, 7, 6, 5}}, // unknown between fmt and data
		w64TestChunk{guid: w64GUID("data"), data: data.Bytes()},
	), samples
}

// pcm8OddFixture is an 8-bit mono PCM WAVE64 file whose data chunk holds
// an odd number of bytes.
func pcm8OddFixture() ([]byte, []int) {
	samples := []int{10, 20, 30, 40, 50}
	data := &bytes.Buffer{}
	for _, s := range samples {
		data.WriteByte(byte(s))
	}
	return buildWave64(
		w64TestChunk{guid: w64GUID("fmt "), data: w64Fmt(1, 1, 22050, 8)},
		w64TestChunk{guid: w64GUID("data"), data: data.Bytes()},
	), samples
}

// float32Fixture is a 32-bit IEEE float stereo WAVE64 file.
func float32Fixture() ([]byte, []int32) {
	samples := []int32{0x00000000, 0x3F000000, -0x41000000, 0x3F800000} // 0.0, 0.5, -0.5, 1.0 as bits
	data := &bytes.Buffer{}
	for _, s := range samples {
		binary.Write(data, binary.LittleEndian, s)
	}
	return buildWave64(
		w64TestChunk{guid: w64GUID("fmt "), data: w64Fmt(3, 2, 48000, 32)},
		w64TestChunk{guid: w64GUID("data"), data: data.Bytes()},
	), samples
}

func TestWave64PCMDecode(t *testing.T) {
	fixture, samples := pcm16Fixture()
	d := NewDecoder(bytes.NewReader(fixture))
	d.ReadInfo()
	if d.Err() != nil {
		t.Fatalf("unexpected error reading headers: %v", d.Err())
	}
	if !d.isWave64 {
		t.Fatal("expected the container to be detected as WAVE64")
	}
	if d.NumChans != 1 || d.BitDepth != 16 || d.SampleRate != 44100 {
		t.Fatalf("unexpected format: %d ch, %d bits, %d Hz", d.NumChans, d.BitDepth, d.SampleRate)
	}
	if d.WavAudioFormat != 1 {
		t.Fatalf("expected PCM format tag 1, got %d", d.WavAudioFormat)
	}
	// 5 samples * 2 bytes
	if d.PCMLen() != 0 && d.PCMLen() != 10 {
		t.Fatalf("unexpected PCM length %d", d.PCMLen())
	}
	if err := d.FwdToPCM(); err != nil {
		t.Fatalf("unexpected error forwarding to PCM: %v", err)
	}
	if d.PCMLen() != 10 {
		t.Fatalf("expected a PCM length of 10 bytes, got %d", d.PCMLen())
	}
	buf, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatalf("unexpected error decoding PCM: %v", err)
	}
	if !reflect.DeepEqual(buf.Data, samples) {
		t.Fatalf("expected samples %v, got %v", samples, buf.Data)
	}
	if buf.SourceBitDepth != 16 {
		t.Fatalf("expected a source bit depth of 16, got %d", buf.SourceBitDepth)
	}
}

func TestWave64OddDataSize(t *testing.T) {
	fixture, samples := pcm8OddFixture()
	d := NewDecoder(bytes.NewReader(fixture))
	d.ReadInfo()
	if d.Err() != nil {
		t.Fatalf("unexpected error reading headers: %v", d.Err())
	}
	if d.BitDepth != 8 {
		t.Fatalf("expected a bit depth of 8, got %d", d.BitDepth)
	}
	if err := d.FwdToPCM(); err != nil {
		t.Fatalf("unexpected error forwarding to PCM: %v", err)
	}
	if d.PCMLen() != 5 {
		t.Fatalf("expected a PCM length of 5 bytes, got %d", d.PCMLen())
	}
	buf, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatalf("unexpected error decoding PCM: %v", err)
	}
	if !reflect.DeepEqual(buf.Data, samples) {
		t.Fatalf("expected samples %v, got %v", samples, buf.Data)
	}
}

func TestWave64FloatDecode(t *testing.T) {
	fixture, samples := float32Fixture()
	d := NewDecoder(bytes.NewReader(fixture))
	d.ReadInfo()
	if d.Err() != nil {
		t.Fatalf("unexpected error reading headers: %v", d.Err())
	}
	if d.WavAudioFormat != 3 {
		t.Fatalf("expected IEEE float format tag 3, got %d", d.WavAudioFormat)
	}
	if d.NumChans != 2 || d.BitDepth != 32 || d.SampleRate != 48000 {
		t.Fatalf("unexpected format: %d ch, %d bits, %d Hz", d.NumChans, d.BitDepth, d.SampleRate)
	}
	buf, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatalf("unexpected error decoding PCM: %v", err)
	}
	if len(buf.Data) != len(samples) {
		t.Fatalf("expected %d samples, got %d", len(samples), len(buf.Data))
	}
	for i, s := range samples {
		if buf.Data[i] != int(s) {
			t.Fatalf("sample %d: expected bits %#x, got %#x", i, s, buf.Data[i])
		}
	}
}

func TestWave64PCMBufferAndRewind(t *testing.T) {
	fixture, samples := pcm16Fixture()
	d := NewDecoder(bytes.NewReader(fixture))
	buf := &audio.IntBuffer{Data: make([]int, 8)}
	n, err := d.PCMBuffer(buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n != len(samples) {
		t.Fatalf("expected %d samples, got %d", len(samples), n)
	}
	if !reflect.DeepEqual(buf.Data[:n], samples) {
		t.Fatalf("expected samples %v, got %v", samples, buf.Data[:n])
	}
	// the buffer is drained, further reads return nothing
	n, err = d.PCMBuffer(buf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n != 0 {
		t.Fatalf("expected 0 samples after draining, got %d", n)
	}
	if err := d.Rewind(); err != nil {
		t.Fatalf("unexpected rewind error: %v", err)
	}
	buf2 := &audio.IntBuffer{Data: make([]int, 8)}
	n, err = d.PCMBuffer(buf2)
	if err != nil {
		t.Fatalf("unexpected error after rewind: %v", err)
	}
	if !reflect.DeepEqual(buf2.Data[:n], samples) {
		t.Fatalf("expected samples %v after rewind, got %v", samples, buf2.Data[:n])
	}
}

func TestWave64DurationAndValidity(t *testing.T) {
	fixture, _ := pcm16Fixture()
	d := NewDecoder(bytes.NewReader(fixture))
	if !d.IsValidFile() {
		t.Fatal("expected the WAVE64 fixture to be valid")
	}
	dur, err := d.Duration()
	if err != nil {
		t.Fatalf("unexpected duration error: %v", err)
	}
	// container size / avg bytes per sec, same approximation as RIFF
	expected := time.Duration(float64(len(fixture)) / float64(44100*2) * float64(time.Second))
	if dur != expected {
		t.Fatalf("expected a duration of %s, got %s", expected, dur)
	}
	if !strings.Contains(d.String(), "44100") {
		t.Fatalf("unexpected String() output: %s", d.String())
	}
}

// errSeekReader wraps a reader and fails all seeks, simulating a
// non-seekable input.
type errSeekReader struct {
	r io.Reader
}

func (e errSeekReader) Read(p []byte) (int, error) { return e.r.Read(p) }
func (e errSeekReader) Seek(int64, int) (int64, error) {
	return 0, errors.New("seek not supported")
}

func TestWave64Errors(t *testing.T) {
	goodPCM, _ := pcm16Fixture()
	fmtChunk := w64TestChunk{guid: w64GUID("fmt "), data: w64Fmt(1, 1, 44100, 16)}
	dataChunk := w64TestChunk{guid: w64GUID("data"), data: []byte{1, 2, 3, 4}}

	badBlockAlign := w64Fmt(1, 2, 44100, 16)
	binary.LittleEndian.PutUint16(badBlockAlign[12:14], 999)

	unsupportedFmt := w64Fmt(0x55, 1, 44100, 16) // mp3

	oversized := buildWave64(w64TestChunk{
		guid:         w64GUID("data"),
		data:         []byte{1, 2, 3, 4},
		sizeOverride: 1 << 40,
	})
	// shrink the declared container size so the chunk overlaps the bounds
	overlap := buildWave64(
		w64TestChunk{guid: w64GUID("fmt "), data: w64Fmt(1, 1, 44100, 16), sizeOverride: 1 << 30},
		dataChunk,
	)

	tinyChunk := buildWave64(w64TestChunk{
		guid:         w64GUID("fmt "),
		data:         nil,
		sizeOverride: 16, // smaller than the 24-byte chunk header
	})

	testCases := []struct {
		name   string
		data   []byte
		errMsg string
	}{
		{"short container header", goodPCM[:10], "failed to read the container header"},
		{"short wave guid", goodPCM[:30], "wave form GUID"},
		{"bad wave guid", append(append([]byte{}, goodPCM[:24]...), []byte("nope0123456789ab")...), "not a WAVE container"},
		{"declared size too small", func() []byte {
			b := buildWave64(fmtChunk, dataChunk)
			binary.LittleEndian.PutUint64(b[16:24], 10)
			return b
		}(), "invalid container size"},
		{"missing fmt", buildWave64(dataChunk), "fmt chunk not found"},
		{"missing data", buildWave64(fmtChunk), "PCM Chunk not found"},
		{"invalid block align", buildWave64(
			w64TestChunk{guid: w64GUID("fmt "), data: badBlockAlign}, dataChunk), "invalid block align"},
		{"unsupported format", buildWave64(
			w64TestChunk{guid: w64GUID("fmt "), data: unsupportedFmt}, dataChunk), "unsupported audio format"},
		{"oversized chunk", oversized, "exceeds the container bounds"},
		{"overlapping chunk", overlap, "exceeds the container bounds"},
		{"chunk smaller than header", tinyChunk, "invalid chunk size"},
		{"truncated chunk header", goodPCM[:50], "short read"},
	}

	for _, tc := range testCases {
		d := NewDecoder(bytes.NewReader([]byte(tc.data)))
		d.ReadInfo()
		err := d.Err()
		if err == nil {
			// some failures only surface when looking for the data chunk
			err = d.FwdToPCM()
		}
		if err == nil {
			t.Fatalf("%s: expected an error containing %q, got nil", tc.name, tc.errMsg)
		}
		if !strings.Contains(err.Error(), tc.errMsg) {
			t.Fatalf("%s: expected an error containing %q, got %q", tc.name, tc.errMsg, err.Error())
		}
	}
}

func TestWave64NonSeekableInput(t *testing.T) {
	fixture, _ := pcm16Fixture()
	d := NewDecoder(errSeekReader{r: bytes.NewReader(fixture)})
	d.ReadInfo()
	if d.Err() == nil || !strings.Contains(d.Err().Error(), "seekable reader") {
		t.Fatalf("expected a seekable reader error, got %v", d.Err())
	}
}

func TestWave64MissingDataPCMBuffer(t *testing.T) {
	fmtOnly := buildWave64(w64TestChunk{guid: w64GUID("fmt "), data: w64Fmt(1, 1, 44100, 16)})
	d := NewDecoder(bytes.NewReader(fmtOnly))
	buf := &audio.IntBuffer{Data: make([]int, 8)}
	_, err := d.PCMBuffer(buf)
	if err == nil || !strings.Contains(err.Error(), "PCM Chunk not found") {
		t.Fatalf("expected a PCM chunk not found error, got %v", err)
	}
}

// TestWave64DoesNotBreakRIFF makes sure a RIFF file is still detected and
// decoded as RIFF after the container auto-detection.
func TestWave64DoesNotBreakRIFF(t *testing.T) {
	// minimal RIFF/WAVE file: 1 channel, 8-bit PCM, 3 samples
	data := []byte{
		'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E',
		'f', 'm', 't', ' ', 16, 0, 0, 0,
		1, 0, 1, 0, 0x44, 0xAC, 0, 0, 0x44, 0xAC, 0, 0, 1, 0, 8, 0,
		'd', 'a', 't', 'a', 3, 0, 0, 0, 10, 20, 30,
	}
	binary.LittleEndian.PutUint32(data[4:8], uint32(len(data)-8))
	d := NewDecoder(bytes.NewReader(data))
	d.ReadInfo()
	if d.Err() != nil {
		t.Fatalf("unexpected error: %v", d.Err())
	}
	if d.isWave64 {
		t.Fatal("RIFF file wrongly detected as WAVE64")
	}
	buf, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(buf.Data, []int{10, 20, 30}) {
		t.Fatalf("unexpected samples: %v", buf.Data)
	}
}

func TestWave64UnknownGUIDTailSkipped(t *testing.T) {
	// an unknown chunk whose GUID doesn't follow the RIFF-derived pattern
	// must be skipped using its 64-bit size and 8-byte alignment.
	samples := []int{7, -7}
	data := &bytes.Buffer{}
	for _, s := range samples {
		binary.Write(data, binary.LittleEndian, int16(s))
	}
	fixture := buildWave64(
		w64TestChunk{guid: [16]byte{0xDE, 0xAD, 0xBE, 0xEF, 0x01}, data: []byte{0x00}},
		w64TestChunk{guid: w64GUID("fmt "), data: w64Fmt(1, 1, 8000, 16)},
		w64TestChunk{guid: w64GUID("data"), data: data.Bytes()},
	)
	d := NewDecoder(bytes.NewReader(fixture))
	buf, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(buf.Data, samples) {
		t.Fatalf("expected samples %v, got %v", samples, buf.Data)
	}
}

func TestWave64OversizedDataLength(t *testing.T) {
	// a data chunk claiming more than MaxInt32 bytes of payload must be
	// rejected even if the declared container size allows it.
	chunk := w64TestChunk{
		guid:         w64GUID("data"),
		data:         nil,
		sizeOverride: uint64(math.MaxInt32) + 1 + wave64ChunkHeaderSize,
	}
	fixture := buildWave64(
		w64TestChunk{guid: w64GUID("fmt "), data: w64Fmt(1, 1, 44100, 16)},
		chunk,
	)
	// grow the declared container size beyond the chunk
	binary.LittleEndian.PutUint64(fixture[16:24], uint64(math.MaxInt32)*4)
	d := NewDecoder(bytes.NewReader(fixture))
	err := d.FwdToPCM()
	if err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("expected a too large error, got %v", err)
	}
}
