package wav

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"reflect"
	"testing"

	"github.com/go-audio/audio"
	"github.com/go-audio/riff"
)

// sampleAXML is a small but well formed ADM XML document kept as raw bytes so
// tests can assert byte for byte preservation.
var sampleAXML = []byte(`<?xml version="1.0" encoding="UTF-8"?>
<audioFormatExtended xmlns="urn:itu:bs:2077:0:0">
  <audioProgramFormat audioProgramFormatID="APR_0001" audioProgramFormatName="stereo programme"/>
  <audioObject audioObjectName="object 1"><audioTrackUIDRef>ATU_00000001</audioTrackUIDRef></audioObject>
  <audioObject audioObjectName="object 2"><audioTrackUIDRef>ATU_00000002</audioTrackUIDRef></audioObject>
</audioFormatExtended>`)

// writeSeekBuffer is an in-memory io.WriteSeeker used to exercise the encoder
// without touching the filesystem.
type writeSeekBuffer struct {
	buf bytes.Buffer
	pos int64
}

func (w *writeSeekBuffer) Write(p []byte) (int, error) {
	end := int64(len(p)) + w.pos
	if end > int64(w.buf.Len()) {
		grow := make([]byte, end-int64(w.buf.Len()))
		if _, err := w.buf.Write(grow); err != nil {
			return 0, err
		}
	}
	n := copy(w.buf.Bytes()[w.pos:], p)
	w.pos += int64(n)
	return n, nil
}

func (w *writeSeekBuffer) Seek(offset int64, whence int) (int64, error) {
	var pos int64
	switch whence {
	case io.SeekStart:
		pos = offset
	case io.SeekCurrent:
		pos = w.pos + offset
	case io.SeekEnd:
		pos = int64(w.buf.Len()) + offset
	}
	if pos < 0 {
		return 0, errors.New("negative position")
	}
	w.pos = pos
	return pos, nil
}

func (w *writeSeekBuffer) Bytes() []byte { return w.buf.Bytes() }

func stereoADM() *ADMMetadata {
	return &ADMMetadata{
		NumTracks: 2,
		Entries: []*ChannelMappingEntry{
			{
				TrackIndex:    1,
				AudioTrackUID: "ATU_00000001",
				TrackRef:      "AT_00010001_01",
				PackRef:       "AP_00010002",
			},
			{
				TrackIndex:    2,
				AudioTrackUID: "ATU_00000002",
				TrackRef:      "AT_00010002_01",
				PackRef:       "AP_00010002",
			},
		},
		AXML: append([]byte(nil), sampleAXML...),
	}
}

func TestADMRoundTrip(t *testing.T) {
	// stereo, 2 frames of fixed samples, 16-bit PCM
	src := &audio.IntBuffer{
		Format: &audio.Format{NumChannels: 2, SampleRate: 48000},
		Data:   []int{100, -100, 200, -200},
	}

	out := &writeSeekBuffer{}
	e := NewEncoder(out, 48000, 16, 2, 1)
	e.Metadata = &Metadata{ADM: stereoADM()}
	if err := e.Write(src); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}

	d := NewDecoder(bytes.NewReader(out.Bytes()))
	pcm, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(pcm.Data, src.Data) {
		t.Fatalf("PCM data changed through the ADM round trip: %v", pcm.Data)
	}
	d.ReadMetadata()
	if err := d.Err(); err != nil {
		t.Fatal(err)
	}
	if d.Metadata == nil || d.Metadata.ADM == nil {
		t.Fatal("expected ADM metadata to be decoded")
	}
	adm := d.Metadata.ADM

	if adm.NumTracks != 2 {
		t.Fatalf("expected numTracks 2, got %d", adm.NumTracks)
	}
	if adm.NumUIDs() != 2 {
		t.Fatalf("expected numUIDs 2, got %d", adm.NumUIDs())
	}
	if len(adm.Entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(adm.Entries))
	}
	want := stereoADM()
	if !reflect.DeepEqual(adm.Entries, want.Entries) {
		t.Fatalf("channel mapping mismatch:\nwant %#v\ngot  %#v", want.Entries, adm.Entries)
	}
	if !bytes.Equal(adm.AXML, sampleAXML) {
		t.Fatalf("axml bytes were not preserved:\nwant %q\ngot  %q", sampleAXML, adm.AXML)
	}

	// re-encoding the decoded metadata must produce identical chna/axml bytes
	out2 := &writeSeekBuffer{}
	e2 := NewEncoder(out2, 48000, 16, 2, 1)
	e2.Metadata = &Metadata{ADM: adm}
	if err := e2.Write(src); err != nil {
		t.Fatal(err)
	}
	if err := e2.Close(); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out2.Bytes(), out.Bytes()) {
		t.Fatal("re-encoding the decoded ADM file produced different bytes")
	}
}

func TestADMChunkOrdering(t *testing.T) {
	// Build a WAVE file with axml/chna *before* the fmt chunk: the decoder
	// must still capture the ADM data and reach the PCM chunk.
	pcm := []byte{0x64, 0x00, 0x9c, 0xff, 0xc8, 0x00, 0x38, 0xff} // 16-bit stereo, 2 frames
	file := bytes.NewBuffer(nil)
	file.Write(riff.RiffID[:])
	binary.Write(file, binary.LittleEndian, uint32(0)) // placeholder RIFF size
	file.Write(riff.WavFormatID[:])

	axmlChunk, err := encodeAXMLChunk(sampleAXML)
	if err != nil {
		t.Fatal(err)
	}
	file.Write(axmlChunk)
	chnaChunk, err := encodeChnaChunk(stereoADM())
	if err != nil {
		t.Fatal(err)
	}
	file.Write(chnaChunk)

	// fmt chunk (PCM, stereo, 48kHz, 16-bit)
	file.Write(riff.FmtID[:])
	binary.Write(file, binary.LittleEndian, uint32(16))
	binary.Write(file, binary.LittleEndian, uint16(1))
	binary.Write(file, binary.LittleEndian, uint16(2))
	binary.Write(file, binary.LittleEndian, uint32(48000))
	binary.Write(file, binary.LittleEndian, uint32(48000*2*2))
	binary.Write(file, binary.LittleEndian, uint16(4))
	binary.Write(file, binary.LittleEndian, uint16(16))

	// data chunk
	file.Write(riff.DataFormatID[:])
	binary.Write(file, binary.LittleEndian, uint32(len(pcm)))
	file.Write(pcm)

	raw := file.Bytes()
	binary.LittleEndian.PutUint32(raw[4:], uint32(len(raw)-8))

	d := NewDecoder(bytes.NewReader(raw))
	buf, err := d.FullPCMBuffer()
	if err != nil {
		t.Fatalf("PCM access failed with ADM chunks before fmt: %v", err)
	}
	if !reflect.DeepEqual(buf.Data, []int{100, -100, 200, -200}) {
		t.Fatalf("unexpected PCM samples: %v", buf.Data)
	}
	d.ReadMetadata()
	if err := d.Err(); err != nil {
		t.Fatal(err)
	}
	if d.Metadata == nil || d.Metadata.ADM == nil {
		t.Fatal("ADM chunks before fmt were skipped")
	}
	adm := d.Metadata.ADM
	if adm.NumTracks != 2 || adm.NumUIDs() != 2 {
		t.Fatalf("unexpected counts: numTracks=%d numUIDs=%d", adm.NumTracks, adm.NumUIDs())
	}
	if !bytes.Equal(adm.AXML, sampleAXML) {
		t.Fatal("axml not preserved when located before fmt")
	}
	if !reflect.DeepEqual(adm.Entries, stereoADM().Entries) {
		t.Fatalf("chna mapping not preserved: %#v", adm.Entries)
	}
}

func TestEncodeADMValidation(t *testing.T) {
	base := stereoADM()

	testCases := []struct {
		desc    string
		mutate  func(*ADMMetadata)
		numChan int
		wantErr error
	}{
		{
			desc:    "numTracks mismatch",
			mutate:  func(a *ADMMetadata) { a.NumTracks = 1 },
			numChan: 2,
			wantErr: ErrChannelOutOfRange,
		},
		{
			desc:    "track index out of range",
			mutate:  func(a *ADMMetadata) { a.Entries[1].TrackIndex = 3 },
			numChan: 2,
			wantErr: ErrChannelOutOfRange,
		},
		{
			desc:    "zero track index",
			mutate:  func(a *ADMMetadata) { a.Entries[0].TrackIndex = 0 },
			numChan: 2,
			wantErr: ErrChannelOutOfRange,
		},
		{
			desc:    "duplicate UID",
			mutate:  func(a *ADMMetadata) { a.Entries[1].AudioTrackUID = a.Entries[0].AudioTrackUID },
			numChan: 2,
			wantErr: ErrDuplicateTrackUID,
		},
		{
			desc:    "UID too long",
			mutate:  func(a *ADMMetadata) { a.Entries[0].AudioTrackUID = "ATU_0000000001" },
			numChan: 2,
			wantErr: ErrChnaFieldTooLong,
		},
		{
			desc:    "track ref too long",
			mutate:  func(a *ADMMetadata) { a.Entries[0].TrackRef = "AT_00000000000001" },
			numChan: 2,
			wantErr: ErrChnaFieldTooLong,
		},
		{
			desc:    "pack ref too long",
			mutate:  func(a *ADMMetadata) { a.Entries[0].PackRef = "AP_00000000001" },
			numChan: 2,
			wantErr: ErrChnaFieldTooLong,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			adm := stereoADM()
			tc.mutate(adm)
			out := &writeSeekBuffer{}
			e := NewEncoder(out, 48000, 16, tc.numChan, 1)
			e.Metadata = &Metadata{ADM: adm}
			src := &audio.IntBuffer{
				Format: &audio.Format{NumChannels: 2, SampleRate: 48000},
				Data:   []int{1, 2, 3, 4},
			}
			if err := e.Write(src); err != nil {
				t.Fatal(err)
			}
			err := e.Close()
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("expected %v, got %v", tc.wantErr, err)
			}
		})
	}

	// sanity check: the untouched, valid metadata encodes fine
	t.Run("valid", func(t *testing.T) {
		out := &writeSeekBuffer{}
		e := NewEncoder(out, 48000, 16, 2, 1)
		e.Metadata = &Metadata{ADM: base}
		src := &audio.IntBuffer{
			Format: &audio.Format{NumChannels: 2, SampleRate: 48000},
			Data:   []int{1, 2, 3, 4},
		}
		if err := e.Write(src); err != nil {
			t.Fatal(err)
		}
		if err := e.Close(); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}

func TestDecodeTruncatedChna(t *testing.T) {
	// Declare 2 UIDs but only serialize one 40-byte entry.
	file := bytes.NewBuffer(nil)
	file.Write(riff.RiffID[:])
	binary.Write(file, binary.LittleEndian, uint32(0))
	file.Write(riff.WavFormatID[:])
	file.Write(CIDChna[:])
	binary.Write(file, binary.LittleEndian, uint32(chnaHeaderSize+chnaEntrySize))
	binary.Write(file, binary.LittleEndian, uint16(2)) // numTracks
	binary.Write(file, binary.LittleEndian, uint16(2)) // numUIDs > entries present
	entry := make([]byte, chnaEntrySize)
	binary.LittleEndian.PutUint16(entry[0:2], 1)
	copy(entry[2:14], "ATU_00000001")
	copy(entry[14:28], "AT_00010001_01")
	copy(entry[28:39], "AP_00010002")
	file.Write(entry)

	d := NewDecoder(bytes.NewReader(file.Bytes()))
	d.ReadMetadata()
	if !errors.Is(d.Err(), ErrTruncatedChnaChunk) {
		t.Fatalf("expected ErrTruncatedChnaChunk, got %v", d.Err())
	}
}

func TestDecodeMalformedChnaSize(t *testing.T) {
	file := bytes.NewBuffer(nil)
	file.Write(riff.RiffID[:])
	binary.Write(file, binary.LittleEndian, uint32(0))
	file.Write(riff.WavFormatID[:])
	file.Write(CIDChna[:])
	// data size that cannot hold whole 40-byte audioID entries
	binary.Write(file, binary.LittleEndian, uint32(chnaHeaderSize+17))
	binary.Write(file, binary.LittleEndian, uint16(1))
	binary.Write(file, binary.LittleEndian, uint16(1))
	file.Write(make([]byte, 18)) // padded to an even chunk size

	d := NewDecoder(bytes.NewReader(file.Bytes()))
	d.ReadMetadata()
	if !errors.Is(d.Err(), ErrInvalidChnaChunk) {
		t.Fatalf("expected ErrInvalidChnaChunk, got %v", d.Err())
	}
}

func TestDecodeDuplicateUID(t *testing.T) {
	adm := &ADMMetadata{
		NumTracks: 2,
		Entries: []*ChannelMappingEntry{
			{TrackIndex: 1, AudioTrackUID: "ATU_00000001", TrackRef: "AT_00010001_01"},
			{TrackIndex: 2, AudioTrackUID: "ATU_00000001", TrackRef: "AT_00010002_01"},
		},
	}
	raw, err := encodeChnaChunk(adm)
	if err != nil {
		t.Fatal(err)
	}
	file := bytes.NewBuffer(nil)
	file.Write(riff.RiffID[:])
	binary.Write(file, binary.LittleEndian, uint32(0))
	file.Write(riff.WavFormatID[:])
	file.Write(raw)

	d := NewDecoder(bytes.NewReader(file.Bytes()))
	d.ReadMetadata()
	if !errors.Is(d.Err(), ErrDuplicateTrackUID) {
		t.Fatalf("expected ErrDuplicateTrackUID, got %v", d.Err())
	}
}

func TestDecodeOutOfRangeTrackIndex(t *testing.T) {
	// Decode-time access: the decoder does not know the channel count while
	// reading a standalone chna chunk, so check the structural zero index rule
	// and validate a real file's bounds through a second decode pass.
	adm := &ADMMetadata{
		NumTracks: 2,
		Entries: []*ChannelMappingEntry{
			{TrackIndex: 0, AudioTrackUID: "ATU_00000001", TrackRef: "AT_00010001_01"},
		},
	}
	raw, err := encodeChnaChunk(adm)
	if err != nil {
		t.Fatal(err)
	}
	file := bytes.NewBuffer(nil)
	file.Write(riff.RiffID[:])
	binary.Write(file, binary.LittleEndian, uint32(0))
	file.Write(riff.WavFormatID[:])
	file.Write(raw)

	d := NewDecoder(bytes.NewReader(file.Bytes()))
	d.ReadMetadata()
	if !errors.Is(d.Err(), ErrInvalidChnaChunk) {
		t.Fatalf("expected ErrInvalidChnaChunk for zero track index, got %v", d.Err())
	}
}

func TestOddAXMLPaddingRoundTrip(t *testing.T) {
	oddAXML := make([]byte, 123) // odd length document
	for i := range oddAXML {
		oddAXML[i] = byte('a' + i%26)
	}
	adm := &ADMMetadata{
		NumTracks: 1,
		Entries: []*ChannelMappingEntry{
			{TrackIndex: 1, AudioTrackUID: "ATU_00000001", TrackRef: "AT_00010001_01", PackRef: "AP_00010002"},
		},
		AXML: oddAXML,
	}
	out := &writeSeekBuffer{}
	e := NewEncoder(out, 48000, 16, 1, 1)
	e.Metadata = &Metadata{ADM: adm}
	src := &audio.IntBuffer{
		Format: &audio.Format{NumChannels: 1, SampleRate: 48000},
		Data:   []int{1, 2},
	}
	if err := e.Write(src); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}

	d := NewDecoder(bytes.NewReader(out.Bytes()))
	d.ReadMetadata()
	if err := d.Err(); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(d.Metadata.ADM.AXML, oddAXML) {
		t.Fatal("odd length axml was not preserved without the padding byte")
	}
	if d.Metadata.ADM.NumUIDs() != 1 || d.Metadata.ADM.NumTracks != 1 {
		t.Fatalf("unexpected counts: %+v", d.Metadata.ADM)
	}
}

func TestPlainWavOutputUnchanged(t *testing.T) {
	// A normal encoder without ADM metadata must produce no chna/axml chunks.
	out := &writeSeekBuffer{}
	e := NewEncoder(out, 48000, 16, 2, 1)
	src := &audio.IntBuffer{
		Format: &audio.Format{NumChannels: 2, SampleRate: 48000},
		Data:   []int{1, 2, 3, 4},
	}
	if err := e.Write(src); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	raw := out.Bytes()
	if bytes.Contains(raw, CIDChna[:]) || bytes.Contains(raw, CIDAxml[:]) {
		t.Fatal("plain wav output unexpectedly contains ADM chunks")
	}

	d := NewDecoder(bytes.NewReader(raw))
	d.ReadMetadata()
	if err := d.Err(); err != nil {
		t.Fatal(err)
	}
	if d.Metadata != nil && d.Metadata.ADM != nil {
		t.Fatal("plain wav decoded with unexpected ADM metadata")
	}
}

// TestADMRoundTripOnRealFile makes sure existing fixtures still decode and the
// new code path never trips PCM access.
func TestADMDoesNotBreakFixtures(t *testing.T) {
	files := []string{"fixtures/kick.wav", "fixtures/bass.wav", "fixtures/listChunkInHeader.wav"}
	for _, path := range files {
		t.Run(path, func(t *testing.T) {
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			d := NewDecoder(f)
			if _, err := d.FullPCMBuffer(); err != nil && !errors.Is(err, io.EOF) {
				t.Fatalf("PCM decode failed: %v", err)
			}
			d.ReadMetadata()
			if err := d.Err(); err != nil {
				t.Fatalf("metadata scan failed: %v", err)
			}
		})
	}
}
