# wav codec

[![GoDoc](https://godoc.org/github.com/go-audio/wav?status.svg)](https://godoc.org/github.com/go-audio/wav)
[![Go Report Card](https://goreportcard.com/badge/github.com/go-audio/wav)](https://goreportcard.com/report/github.com/go-audio/wav)
[![Coverage Status](https://codecov.io/gh/go-audio/wav/graph/badge.svg)](https://codecov.io/gh/go-audio/wav)
[![Build Status](https://travis-ci.org/go-audio/wav.svg)](https://travis-ci.org/go-audio/wav)

Wav audio file encoder and decoder. See [GoDoc](https://godoc.org/github.com/go-audio/wav) for more details.

## Sony Wave64 (.w64)

Classic RIFF/WAVE files store chunk sizes in 32 bit fields and therefore cannot
hold more than 4 GiB of audio. For long recordings use `NewWave64Encoder`,
which writes a Sony Wave64 container: 16 byte GUIDs replace the FourCCs and
every chunk size is a 64 bit value.

```go
e, err := wav.NewWave64Encoder(out, sampleRate, bitDepth, numChans)
```

The Wave64 PCM mode supports 16 or 24 bit samples in mono or stereo. Chunks are
8 byte aligned (zero padding is added when needed), and chunk sizes include
their own 24 byte header as required by the format. `NewEncoder` keeps writing
the classic, byte-for-byte unchanged RIFF/WAVE layout. Use `Encoder.IsWave64`
to tell which container an encoder targets.
