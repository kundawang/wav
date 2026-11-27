package wav

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"

	"github.com/go-audio/riff"
)

// ADM (Audio Definition Model) metadata is carried by two RIFF chunks:
//   - the "chna" chunk maps the interleaved audio tracks of the data chunk
//     to ADM audioTrackUIDs through track and pack format references;
//   - the "axml" chunk carries the raw, XML serialized ADM document.
//
// The on-disk layout of a chna chunk is defined by EBU Tech 3364
// Supplement 7 / ITU-R BS.2088:
//
//	numTracks            uint16 LE
//	numUIDs              uint16 LE
//	entries[numUIDs]     audioID, 40 bytes each:
//	  trackIndex         uint16 LE (1-based index of the track in the data chunk)
//	  audioTrackUID      12 bytes, NUL padded ASCII
//	  audioTrackFormatID 14 bytes, NUL padded ASCII
//	  audioPackFormatID  11 bytes, NUL padded ASCII
//	  pad                1 NUL byte keeping the entry even sized
var (
	// CIDChna is the chunk ID of the ADM channel mapping chunk.
	CIDChna = [4]byte{'c', 'h', 'n', 'a'}
	// CIDAxml is the chunk ID of the ADM XML metadata chunk.
	CIDAxml = [4]byte{'a', 'x', 'm', 'l'}
)

const (
	chnaHeaderSize     = 4 // numTracks (uint16) + numUIDs (uint16)
	chnaEntrySize      = 40
	chnaTrackIndexSize = 2
	chnaUIDSize        = 12
	chnaTrackRefSize   = 14
	chnaPackRefSize    = 11
	chnaPadSize        = 1
)

var (
	// ErrTruncatedChnaChunk is returned when a chna chunk does not contain
	// enough bytes for the advertised number of audioID entries.
	ErrTruncatedChnaChunk = errors.New("truncated chna chunk")
	// ErrInvalidChnaChunk is returned when the size of a chna chunk cannot be
	// parsed into fixed length audioID entries.
	ErrInvalidChnaChunk = errors.New("invalid chna chunk size")
	// ErrChannelOutOfRange is returned when a chna entry references a track
	// index that does not exist in the file.
	ErrChannelOutOfRange = errors.New("chna track index out of range")
	// ErrDuplicateTrackUID is returned when the same audioTrackUID is used by
	// more than one chna entry.
	ErrDuplicateTrackUID = errors.New("duplicate audioTrackUID in chna chunk")
	// ErrChnaFieldTooLong is returned when a fixed length chna string field
	// cannot hold the provided value.
	ErrChnaFieldTooLong = errors.New("chna string field too long")
)

// ADMMetadata holds the restricted ADM channel mapping information of a wav
// file. AXML is preserved byte for byte: the package never parses or rewrites
// the XML document.
type ADMMetadata struct {
	// NumTracks is the number of audio tracks (channels) declared by the chna
	// chunk. On write it must match the number of channels of the encoder.
	NumTracks uint16
	// Entries is the list of audioID entries, one per audioTrackUID.
	Entries []*ChannelMappingEntry
	// AXML is the raw content of the axml chunk, preserved as read.
	AXML []byte
}

// NumUIDs returns the number of audioTrackUID entries in the channel mapping.
func (m *ADMMetadata) NumUIDs() uint16 {
	if m == nil {
		return 0
	}
	return uint16(len(m.Entries))
}

// ChannelMappingEntry is a single 40 bytes audioID record of a chna chunk.
type ChannelMappingEntry struct {
	// TrackIndex is the 1-based index of the track in the interleaved data
	// chunk this entry applies to.
	TrackIndex uint16
	// AudioTrackUID is the ADM audioTrackUID reference (e.g. ATU_00000001).
	AudioTrackUID string
	// TrackRef is the audioTrackFormatID reference (e.g. AT_00010001_01).
	TrackRef string
	// PackRef is the audioPackFormatID reference (e.g. AP_00010002). It may be
	// empty when no pack reference is required.
	PackRef string
}

// DecodeChnaChunk decodes a chna chunk into d.Metadata.ADM.
func DecodeChnaChunk(d *Decoder, ch *riff.Chunk) error {
	if ch == nil {
		return fmt.Errorf("can't decode a nil chunk")
	}
	if d == nil {
		return fmt.Errorf("nil decoder")
	}
	if ch.ID != CIDChna {
		ch.Drain()
		return nil
	}

	dataSize, err := rawChunkSize(d, ch)
	if err != nil {
		ch.Drain()
		return fmt.Errorf("failed to read the chna chunk header - %w", err)
	}
	adm, err := parseChna(ch, dataSize)
	if err != nil {
		ch.Drain()
		return err
	}
	if d.Metadata == nil {
		d.Metadata = &Metadata{}
	}
	if d.Metadata.ADM == nil {
		d.Metadata.ADM = adm
	} else {
		d.Metadata.ADM.NumTracks = adm.NumTracks
		d.Metadata.ADM.Entries = adm.Entries
	}
	ch.Drain()
	return nil
}

// DecodeAXMLChunk reads the raw axml chunk and stores it, untouched, in
// d.Metadata.ADM.AXML.
func DecodeAXMLChunk(d *Decoder, ch *riff.Chunk) error {
	if ch == nil {
		return fmt.Errorf("can't decode a nil chunk")
	}
	if d == nil {
		return fmt.Errorf("nil decoder")
	}
	if ch.ID != CIDAxml {
		ch.Drain()
		return nil
	}

	// Use the exact size declared in the chunk header. The riff layer rounds
	// odd chunk sizes up to the next word boundary, which would otherwise make
	// the RIFF padding byte look like part of the XML document.
	dataSize, err := rawChunkSize(d, ch)
	if err != nil {
		ch.Drain()
		return fmt.Errorf("failed to read the axml chunk header - %w", err)
	}
	buf := make([]byte, dataSize)
	n, err := io.ReadFull(ch, buf)
	if err != nil {
		ch.Drain()
		return fmt.Errorf("failed to read the axml chunk - %w", err)
	}
	if n != dataSize {
		ch.Drain()
		return fmt.Errorf("failed to read the full axml chunk")
	}
	if d.Metadata == nil {
		d.Metadata = &Metadata{}
	}
	if d.Metadata.ADM == nil {
		d.Metadata.ADM = &ADMMetadata{}
	}
	d.Metadata.ADM.AXML = append([]byte(nil), buf...)
	ch.Drain()
	return nil
}

// rawChunkSize returns the exact data size declared in the on-disk chunk
// header. The riff parser rounds odd sizes up for word alignment, so the
// declared value is recovered by seeking back to the chunk header when the
// underlying reader supports seeking. The rounded chunk size is returned
// unchanged otherwise.
func rawChunkSize(d *Decoder, ch *riff.Chunk) (int, error) {
	if d == nil || d.r == nil || ch.Pos != 0 {
		return ch.Size, nil
	}
	current, err := d.r.Seek(0, io.SeekCurrent)
	if err != nil {
		return ch.Size, nil
	}
	// The reader currently points at the first data byte: the 8 bytes chunk
	// header (4 bytes ID + 4 bytes size) immediately precede it.
	if _, err := d.r.Seek(current-8, io.SeekStart); err != nil {
		return ch.Size, nil
	}
	var header [8]byte
	if _, err := io.ReadFull(d.r, header[:]); err != nil {
		return 0, err
	}
	if _, err := d.r.Seek(current, io.SeekStart); err != nil {
		return 0, err
	}
	return int(binary.LittleEndian.Uint32(header[4:])), nil
}

// parseChna reads and validates the fixed length records of a chna chunk.
func parseChna(ch *riff.Chunk, dataSize int) (*ADMMetadata, error) {
	if dataSize < chnaHeaderSize {
		return nil, fmt.Errorf("%w: chunk too small for the chna header", ErrInvalidChnaChunk)
	}
	if (dataSize-chnaHeaderSize)%chnaEntrySize != 0 {
		return nil, fmt.Errorf("%w: %d bytes do not hold whole audioID entries", ErrInvalidChnaChunk, dataSize)
	}

	buf := make([]byte, dataSize)
	if _, err := io.ReadFull(ch, buf); err != nil {
		return nil, fmt.Errorf("failed to read the chna chunk - %w", err)
	}
	r := bytes.NewReader(buf)

	adm := &ADMMetadata{}
	if err := binary.Read(r, binary.LittleEndian, &adm.NumTracks); err != nil {
		return nil, fmt.Errorf("failed to read chna numTracks - %w", err)
	}
	var numUIDs uint16
	if err := binary.Read(r, binary.LittleEndian, &numUIDs); err != nil {
		return nil, fmt.Errorf("failed to read chna numUIDs - %w", err)
	}

	entryCount := (dataSize - chnaHeaderSize) / chnaEntrySize
	if int(numUIDs) > entryCount {
		return nil, fmt.Errorf("%w: header advertises %d UIDs but only %d entries fit", ErrTruncatedChnaChunk, numUIDs, entryCount)
	}

	record := make([]byte, chnaEntrySize)
	seenUIDs := make(map[string]struct{}, numUIDs)
	for i := 0; i < int(numUIDs); i++ {
		if _, err := r.Read(record); err != nil {
			return nil, fmt.Errorf("%w: missing audioID entry %d - %v", ErrTruncatedChnaChunk, i, err)
		}
		entry := &ChannelMappingEntry{
			TrackIndex:    binary.LittleEndian.Uint16(record[0:2]),
			AudioTrackUID: nullTermStr(record[2:14]),
			TrackRef:      nullTermStr(record[14:28]),
			PackRef:       nullTermStr(record[28:39]),
		}
		if entry.TrackIndex == 0 {
			return nil, fmt.Errorf("%w: entry %d has a zero track index", ErrInvalidChnaChunk, i)
		}
		if _, dup := seenUIDs[entry.AudioTrackUID]; dup {
			return nil, fmt.Errorf("%w: %q", ErrDuplicateTrackUID, entry.AudioTrackUID)
		}
		seenUIDs[entry.AudioTrackUID] = struct{}{}
		adm.Entries = append(adm.Entries, entry)
	}

	return adm, nil
}

// validateADM runs the write-time checks against the encoder configuration.
func validateADM(adm *ADMMetadata, numChans int) error {
	if adm == nil {
		return nil
	}
	if numChans < 0 || int(adm.NumTracks) != numChans {
		return fmt.Errorf("%w: chna numTracks is %d but the file has %d channels", ErrChannelOutOfRange, adm.NumTracks, numChans)
	}
	if len(adm.Entries) > 65535 {
		return fmt.Errorf("%w: %d entries do not fit in the uint16 numUIDs field", ErrInvalidChnaChunk, len(adm.Entries))
	}
	seenUIDs := make(map[string]struct{}, len(adm.Entries))
	for i, entry := range adm.Entries {
		if entry == nil {
			return fmt.Errorf("%w: chna entry %d is nil", ErrInvalidChnaChunk, i)
		}
		if entry.TrackIndex < 1 || int(entry.TrackIndex) > numChans {
			return fmt.Errorf("%w: entry %d references track %d in a %d channel file", ErrChannelOutOfRange, i, entry.TrackIndex, numChans)
		}
		if len(entry.AudioTrackUID) > chnaUIDSize {
			return fmt.Errorf("%w: audioTrackUID %q exceeds %d bytes", ErrChnaFieldTooLong, entry.AudioTrackUID, chnaUIDSize)
		}
		if len(entry.TrackRef) > chnaTrackRefSize {
			return fmt.Errorf("%w: audioTrackFormatID %q exceeds %d bytes", ErrChnaFieldTooLong, entry.TrackRef, chnaTrackRefSize)
		}
		if len(entry.PackRef) > chnaPackRefSize {
			return fmt.Errorf("%w: audioPackFormatID %q exceeds %d bytes", ErrChnaFieldTooLong, entry.PackRef, chnaPackRefSize)
		}
		if _, dup := seenUIDs[entry.AudioTrackUID]; dup {
			return fmt.Errorf("%w: %q", ErrDuplicateTrackUID, entry.AudioTrackUID)
		}
		seenUIDs[entry.AudioTrackUID] = struct{}{}
	}
	return nil
}

// encodeChnaChunk serializes the chna chunk including its RIFF header.
func encodeChnaChunk(adm *ADMMetadata) ([]byte, error) {
	dataSize := chnaHeaderSize + len(adm.Entries)*chnaEntrySize
	buf := bytes.NewBuffer(make([]byte, 0, dataSize+8))

	if err := writeChunkID(buf, CIDChna); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint32(dataSize)); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, adm.NumTracks); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint16(len(adm.Entries))); err != nil {
		return nil, err
	}

	record := make([]byte, chnaEntrySize)
	for _, entry := range adm.Entries {
		for i := range record {
			record[i] = 0
		}
		binary.LittleEndian.PutUint16(record[0:2], entry.TrackIndex)
		copy(record[2:14], entry.AudioTrackUID)
		copy(record[14:28], entry.TrackRef)
		copy(record[28:39], entry.PackRef)
		// record[39] stays NUL: the per-entry even byte padding.
		if _, err := buf.Write(record); err != nil {
			return nil, err
		}
	}

	return buf.Bytes(), nil
}

// encodeAXMLChunk serializes the axml chunk, adding a single NUL padding byte
// when the XML document has an odd length to keep the RIFF chunk word aligned.
func encodeAXMLChunk(axml []byte) ([]byte, error) {
	buf := bytes.NewBuffer(make([]byte, 0, len(axml)+9))
	if err := writeChunkID(buf, CIDAxml); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint32(len(axml))); err != nil {
		return nil, err
	}
	if _, err := buf.Write(axml); err != nil {
		return nil, err
	}
	if len(axml)%2 == 1 {
		if err := buf.WriteByte(0); err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

func writeChunkID(buf *bytes.Buffer, id [4]byte) error {
	_, err := buf.Write(id[:])
	return err
}
