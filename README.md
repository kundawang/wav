# wav codec

[![GoDoc](https://godoc.org/github.com/go-audio/wav?status.svg)](https://godoc.org/github.com/go-audio/wav)
[![Go Report Card](https://goreportcard.com/badge/github.com/go-audio/wav)](https://goreportcard.com/report/github.com/go-audio/wav)
[![Coverage Status](https://codecov.io/gh/go-audio/wav/graph/badge.svg)](https://codecov.io/gh/go-audio/wav)
[![Build Status](https://travis-ci.org/go-audio/wav.svg)](https://travis-ci.org/go-audio/wav)

Wav audio file encoder and decoder. See [GoDoc](https://godoc.org/github.com/go-audio/wav) for more details.

## Streaming decode

`NewDecoder` needs an `io.ReadSeeker` because it may rewind while walking the
chunks. When the audio arrives as a plain forward-only byte stream (an HTTP
body feeding a web player, a pipe, a socket, ...) use `NewStreamDecoder`
instead. It parses RIFF/WAVE one chunk at a time, skips unknown chunks and the
padding byte of odd-sized chunks, and returns PCM samples as soon as they have
been read: nothing past the current buffer is buffered, so playback can start
before the source has finished sending.

```go
d := wav.NewStreamDecoder(resp.Body) // resp.Body is an io.Reader
buf := &audio.IntBuffer{Data: make([]int, 4096)}
for {
	n, err := d.PCMBuffer(buf)
	if err != nil {
		break
	}
	if n == 0 {
		break // end of the data chunk
	}
	play(buf.Data[:n]) // interleaved samples, same values as Decoder
}
```

A `StreamDecoder` only moves forward: `Seek` and `Rewind` return
`ErrStreamNotSeekable`, so open the stream again to replay it. A data chunk
size of `0xFFFFFFFF` (size not known when the header was written) is honoured
and read until EOF. The seekable `NewDecoder` is unchanged and still accepts
`io.ReadSeeker`.

