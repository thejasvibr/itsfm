![](measure_CF_calls_logo.png


# measure_horseshoe_bat_calls
A package to automate measurements of Horseshoe bat calls - (or any similar sound!).

### What is a horseshoe bat call?
Horseshoe bats are a group of echolocating bat that emit calls that look like full or half staple pins. The logo above is a schematic spectrogram of a staplepin type call with the associated measures this package provides. 

This 'staplepin' can be divided into two parts: the constant frequency (CF) and frequency modulated (FM). The CF part is the 'flat' part of the staplepin, where the frequency emitted is constant. The FM parts of the call are where the frequency is either rising and/or descending. Not all calls have both rising and descending. 

### What does this package do?
Given a CF bat call, it segments out CF and FM components and calculates various measures. The FM and CF parts are separated in the waveform and various measurements are made on them including:

* CF peak frequency
* FM end frequency/ies
* CF and FM durations
* CF and FM rms and energy

### Who is this package useful for?
Those interested in studying the structure of bat CF calls - or any sound that looks like a bat CF call?

### Why did you develop this package?
Measuring the sub-structure of horseshoe bat calls can be tiresome and error-prone when done manually. Moreover, there weren't any existing tools specifically developed to handle measurements for horseshoe bat calls. 

This package was developed to automate measurements of *Rhinolophus mehelyi* and *Rhinolophus euryale* bats as they flew in and out of an evening roosting spot. 

 -- Thejasvi Beleyur (thejasvib@gmail.com, tbeleyur@orn.mpg.de)


