# Rando Convolver

Use gradient descent to find an IIR to convolve against to convert 1 sound
to the other through convolution

# Purpose

Sonify gradient descent.

# Difficulties

To calculate the gradient you need N^2 values and thus it doesn't fit
in memory? 9600 x 9600 uses 7gb of video memory

# Notes
- There are clicks on each concatenation. 
  - Maybe we should just make a snippet concatenator that windows the samples
    and overlaps them?

# Tasks
- [ ] Windowed overlap of samples
- [ ] Overlap composition of concatenated ones. 
      Use about 25% overlap on each end?
      Maybe apply an Envelope
      ` ___ `
      `/   \`
- [ ] Choose a good start sound, and a good end sound.
- [ ] Test lower sampling rates (24k, 12k)

# Ideas for later

Should just use deep learning to find the IRR.

Pretty easy to generate examples.
