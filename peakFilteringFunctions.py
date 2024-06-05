# Importing libraries 
import numpy as np
from scipy.spatial import KDTree
from aRead import readAREPO
from tqdm import tqdm

# Function to bin potentials so we can compare to averages
def binPotentials(data, nBins=100):
    # Find all of the potential peaks
    peaks = np.where(data.maxPotential == 1)[0]

    # Calculate cloud CoP
    copx = np.sum(data.potential * data.x) / np.sum(data.potential)
    copy = np.sum(data.potential * data.y) / np.sum(data.potential)
    copz = np.sum(data.potential * data.z) / np.sum(data.potential)

    # Calculate all distances to this
    r = np.sqrt((data.x - copx)**2 + (data.y - copy)**2 + (data.z - copz)**2)
    rPeaks = r[peaks]

    # Bin the distances
    rBins = 10**np.linspace(np.min(np.log10(r)), np.max(np.log10(r)), nBins)
    rBinOld = 0

    # Arrays to store potential information
    averageAtPeak = np.zeros_like(peaks)
    spreadAtPeak = np.zeros_like(peaks)

    # Loop through each bin
    for i in range(len(rBins)):
        # Select potentials inside range
        inRange = np.where((r > rBinOld) & (r < rBins[i]))
        inRangePeaks = np.where((rPeaks > rBinOld) & (rPeaks < rBins[i]))

        # Average and std potential
        average = np.mean(-1 *data.potential[inRange])
        std = np.std(-1 * data.potential[inRange])

        # Assign to peaks
        averageAtPeak[inRangePeaks] = average
        spreadAtPeak[inRangePeaks] = std

        # Update bin
        rBinOld = rBins[i]

    return averageAtPeak, spreadAtPeak

# Filter peaks based on chosen criteria
def allFilters(data, jeansOnly=False, jeansMax=False, jeansMass=False, densityThreshold=0, divAccelerations=False, aboveSigma=0, tenPercent=False):
    # Find all of the potential peaks
    peaks = np.where(data.maxPotential == 1)[0]

    # Construct a tree with all the position data
    tree = KDTree(np.array([data.x, data.y, data.z]).T)

    # Make a tree of only the peaks
    peakOnlyTree = KDTree(np.array([data.x[peaks], data.y[peaks], data.z[peaks]]).T)

    # Array to store if peaks pass all tests
    passTests = np.ones(len(peaks))

    # Storing variables about the peaks
    peakPosition = np.array([data.x[peaks], data.y[peaks], data.z[peaks]]).T
    peakDensity = data.rho[peaks]
    peakTemperature = data.gasTemp[peaks]

    # Work out some info about the peaks
    jeansLengthAtPeak = np.sqrt((15 * 1.38e-16 * peakTemperature) / (4 * np.pi * 6.67e-8 * peakDensity * 2.4 * 1.67e-24))
    jeansMassAtPeak = (4 * np.pi / 3) * (jeansLengthAtPeak**3) * peakDensity

    # Find average potentials and spread
    if aboveSigma != 0:
        averageAtPeak, spreadAtPeak = binPotentials(data)

    # Loop through all the peaks
    for i in range(len(peaks)):
        # Get the current peak ID
        peak = peaks[i]

        # Get all the particles and peaks that are inside this peak's Jeans length
        inJeansLengthPart = tree.query_ball_point(peakPosition[i], jeansLengthAtPeak[i])
        inJeansLengthPeak = peakOnlyTree.query_ball_point(peakPosition[i], jeansLengthAtPeak[i])
        
        ################################
        # No peaks inside jeans length #
        ################################
        if jeansOnly:
            if len(inJeansLengthPeak) > 1:
                passTests[i] = 0

        ####################################
        # Deepest peak inside jeans length #
        ####################################
        peaksInJeansLengthPotential = data.potential[inJeansLengthPeak]
        thisPeakPotential = data.potential[peak]
        if jeansMax:
            if np.any(peaksInJeansLengthPotential < thisPeakPotential):
                passTests[i] = 0

        ###############################################
        # At least one jeans mass inside jeans length #
        ###############################################
        if jeansMass:
            if np.sum(data.mass[inJeansLengthPart]) < jeansMassAtPeak[i]:
                passTests[i] = 0

        ###################################
        # Above a given density threshold #
        ###################################
        if peakDensity[i] < densityThreshold:
            passTests[i] = 0

        ##################################################################### 
        # Negative acceleration divergence of particles inside jeans length #
        #####################################################################
        
        # Remove the peak from the peak neighbours list
        peakIndex = np.where(inJeansLengthPart == peak)
        peakNeighbours = np.delete(inJeansLengthPart, peakIndex)
        
        if divAccelerations:
            # Getting distances of particles to peak
            x = data.x[peakNeighbours] - data.x[peak]
            y = data.y[peakNeighbours] - data.y[peak]
            z = data.z[peakNeighbours] - data.z[peak]

            # Getting relative accelerations
            ax = data.ax[peakNeighbours] - data.ax[peak]
            ay = data.ay[peakNeighbours] - data.ay[peak]
            az = data.az[peakNeighbours] - data.az[peak]

            # Get masses of the particles
            m = data.mass[peakNeighbours]

            # Calculating mass weighted divergence
            dax = ax / x
            day = ay / y
            daz = az / z
            d = dax + day + daz
            div = np.sum(d * m) / np.sum(m)

            if div > 0:
                passTests[i] = 0
            
        ########################################
        # Above sigma of the average potential #
        ########################################
        if aboveSigma != 0:
            if -1 * data.potential[peak] < (averageAtPeak[i] + aboveSigma*spreadAtPeak[i]):
                passTests[i] = 0

        ############################################
        # Above 10% of the deepest other potential #
        ############################################
            
        if tenPercent:
            # Find next biggest peak
            nextDeepest = np.max(data.potential[peakNeighbours])
            thisPlusTen = 1.1 * thisPeakPotential

            if thisPlusTen > nextDeepest:
                passTests[i] = 0
                
    return passTests, jeansLengthAtPeak

def getAveragePeakDistMultiSnap(filepath=" ", filebase="cloudUV1_0", numSnaps=20, skip=1):
   # Create snapshot names
    snapNames = []

    for i in range(25, numSnaps):
        j = i*skip
        if j == 0:
            pass
        else:
            if j < 10:
                snapNames.append(filebase + "0" + str(j) + ".hdf5")
            else:
                snapNames.append(filebase + str(j) + ".hdf5")

    # Create arrays for the stored data
    allDist = []
    allDense = []
    time = []
    averageDist = []
    nPeaks = []

    # Loop through each snap
    for snap in tqdm(snapNames):
        # Load the data 
        data = readAREPO(filepath + snap)

        # Find the peaks
        passTests = allFilters(data, jeansOnly=True, tenPercent=True, aboveSigma=1)
        peaks = np.where(data.maxPotential==1)[0]
        truePeaks = np.where(passTests == 1)
        peaks = peaks[truePeaks]

        # Extract peaks
        x = data.x[peaks]
        y = data.y[peaks]
        z = data.z[peaks]
        rho = data.rho[peaks]

        # Create a KDTree with the peak data
        positions = np.array([x, y, z]).T
        tree = KDTree(positions)

        # Find nearest neighbours for all the peaks
        nearest = tree.query(positions, 2)

        if len(nearest[0]) < 2:
            averageDist = np.append(averageDist, 0)
        else:
            # Get the nearest neighbour distance
            neighbours = nearest[0][:,1]
            dense = nearest[1][:,1]
            
            allDist = np.append(allDist, neighbours)
            allDense = np.append(allDense, rho[dense])
            averageDist = np.append(averageDist, np.mean(neighbours))

        time = np.append(time, data.time)
        nPeaks = np.append(nPeaks, len(peaks))

    return allDist, allDense, time, averageDist, nPeaks