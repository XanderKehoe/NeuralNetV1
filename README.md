# NeuralNetV1
Python neural network class with back propagation

I plan to make another version of this in the future. Although this works, I want to make a new version for two reasons:
* This version does not use matricies for computations, which would be a significant speed boost and make the code more readable. I didn't want to do at first due to fear that I might just end of copying/pasting other code while doing research, I figured this would be the best way to ensure I actually understand the concept.
* This version lacks support for biases

# Brute Force Best Neural Net Architecture Finder
Added a new function to find the best neural net achitecture for a problem by testing out each possibility numerous times in a given range, this approach is time-consuming yet effective, utitilizes multiprocessing to achieve maximum CPU utilization.
