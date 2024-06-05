import numpy as np
import matplotlib.pyplot as plt
import constants as c

class coreProfiles():
    def __init__(self, data, centre, radiusCut=False, radiusValue=0):
        # Assign properties of the data to this object
        self.x = data.x
        self.y = data.y
        self.z = data.z

        self.vx = data.vx
        self.vy = data.vy
        self.vz = data.vz

        self.mass = data.mass
        self.temp = data.gasTemp
        self.rho = data.rho

        # Get the properties of the centre 
        self.xc = self.x[centre]
        self.yc = self.y[centre]
        self.zc = self.z[centre]

        self.vxc = self.vx[centre]
        self.vyc = self.vy[centre]
        self.vzc = self.vz[centre]

        # Find each particle's distance from the centre
        self.rCentre = np.sqrt((self.x - self.xc)**2 + (self.y - self.yc)**2 + (self.z - self.zc)**2)

        if radiusCut:
            self.x = self.x[self.rCentre < radiusValue]
            self.y = self.y[self.rCentre < radiusValue]
            self.z = self.z[self.rCentre < radiusValue]

            self.vx = self.vx[self.rCentre < radiusValue]
            self.vy = self.vy[self.rCentre < radiusValue]
            self.vz = self.vz[self.rCentre < radiusValue]

            self.mass = self.mass[self.rCentre < radiusValue]
            self.temp = self.temp[self.rCentre < radiusValue]
            self.rho = self.rho[self.rCentre < radiusValue]

            self.rCentre = self.rCentre[self.rCentre < radiusValue]

    # Create shell boundaires to work out values within
    def sphericalShells(self, nShell=100):
        # Get logarithimic shell radii
        logR = np.log10(self.rCentre[self.rCentre != 0])

        dShell = (np.max(logR) - np.min(logR)) / nShell
        shellR = np.arange(np.min(logR), np.max(logR), dShell)

        # Assign the boundaries of our shells
        self.shellMin = 10**(shellR - dShell)
        self.shellMin[0] = 0
        self.shellMax = 10**(shellR)

    def profiles(self, nShells=100, velocity=True):
        # Create shells
        self.sphericalShells(nShells)

        # Create arrays
        self.enclosedMass = np.zeros(nShells)
        self.shellMass = np.zeros(nShells)
        self.shellVolume = np.zeros(nShells)
        self.shellDensity = np.zeros(nShells)
        self.shellTemperature = np.zeros(nShells)
        self.shellOpticalDepthIn = np.zeros(nShells)
        self.shellOpticalDepthOut = np.zeros(nShells)
        self.shellOpacity = np.zeros(nShells)

        if velocity:
            self.shellRadialVelocity = np.zeros(nShells)
            self.shellRotationalVelocity = np.zeros(nShells)
            self.shellRadialMach = np.zeros(nShells)
            self.shellRotationalMach = np.zeros(nShells)
            self.velocityProfiles()
        
        # Loop through each shell
        for i in range(nShells):
            # Find particles inside this shell boundary
            belowShell = np.where((self.rCentre < self.shellMax[i]))
            inShell = np.where((self.rCentre < self.shellMax[i]) & (self.rCentre > self.shellMin[i]))

            # Work out the enclosed mass
            thisShellMass = np.sum(self.mass[belowShell])
            self.enclosedMass[i] = thisShellMass

            # Work out the radial density
            self.shellMass[i] = np.sum(self.mass[inShell])
            self.shellVolume[i] = 4 * np.pi * (self.shellMax[i] - self.shellMin[i]) * self.shellMax[i]**2
            self.shellDensity[i] = self.shellMass[i] / self.shellVolume[i]

            # Work out radial temperature
            self.shellTemperature[i] = np.mean(self.temp[inShell])

            # Work out opacity 
            if self.shellTemperature[i] <= 150:
                self.shellOpacity[i] = 2e-4 * self.shellTemperature[i]**2
            else:
                self.shellOpacity[i] = 0.367 * np.sqrt(self.shellTemperature[i])

            # Then work out optical depth
            columnIn = 0
            j = 0
            while j <= i:
                dShell = self.shellMax[j] - self.shellMin[j]
                columnIn += dShell * self.shellDensity[j]
                j += 1

            j = i
            columnOut = 0
            while j < nShells:
                dShell = self.shellMax[j] - self.shellMin[j]
                columnOut += dShell * self.shellDensity[j]
                j += 1
            
            self.shellOpticalDepthIn[i] = columnIn * self.shellOpacity[i]
            self.shellOpticalDepthOut[i] = columnOut * self.shellOpacity[i]

            if velocity:
                self.shellRadialVelocity[i] = np.mean(self.vRad[inShell])
                self.shellRotationalVelocity[i] = np.mean(self.vRot[inShell])

                # Work out the sound speed of this shell
                cs = np.sqrt(1.38e-16 * self.shellTemperature[i] / (2.4 * 1.66e-24))

                # Find the mach number of the rotational and radial velocities
                self.shellRadialMach[i] = self.shellRadialVelocity[i] / cs
                self.shellRotationalMach[i] = self.shellRotationalVelocity[i] / cs

        # Provide an average shell radius to plot against
        self.radius = (self.shellMax + self.shellMin)/2

    def velocityProfiles(self):
        # Calculate the centre of velocity 
        vxC = np.sum(self.vx * self.mass) / np.sum(self.mass)
        vyC = np.sum(self.vy * self.mass) / np.sum(self.mass)
        vzC = np.sum(self.vz * self.mass) / np.sum(self.mass)

        # Subtract central velocity from the particle velocities
        vx = self.vx - vxC
        vy = self.vy - vyC
        vz = self.vz - vzC

        # Work out components
        dx = self.x - self.xc
        dy = self.y - self.yc
        dz = self.z - self.zc

        dvx = vx - self.vxc
        dvy = vy - self.vyc
        dvz = vz - self.vzc

        # Find radial velocity
        self.vRad = (dx * dvx + dy * dvy + dz * dvz) / np.sqrt(dx**2 + dy**2 + dz**2)

        # Find rotational velocity
        self.vRot = ((dy * dvz - dz * dvy) - (dx * dvz - dz * dvx) + (dx * dvy - dy * dvx)) / np.sqrt(dx**2 + dy**2 + dz**2)

    def energyProfiles(self):
        # Sort the particles by their distance to the centre
        sortByRadius = np.argsort(self.rCentre)
        self.rCentreSorted = self.rCentre[sortByRadius]

        # Setup arrays
        self.eGrav = np.zeros(len(sortByRadius-1))
        self.mEnc = np.zeros(len(sortByRadius-1))
        self.eTherm = np.zeros(len(sortByRadius-1))
        self.eKin = np.zeros(len(sortByRadius-1))

        # Assign initial values
        iInit = sortByRadius[0]

        self.mEnc[0] = self.mass[iInit]
        self.eGrav[0] = 6.67e-8 * self.mass[iInit]**2 / (self.mass[iInit] / self.rho[iInit]**(1/3))
        self.eGravTotal = self.eGrav[0]
        self.eTherm[0] = (3/2) * 1.38e-16 * self.temp[iInit] * (self.mass[iInit] / (2.38 * 1.66e-24))

        # Loop through all the particles
        for i in range(1, len(sortByRadius)):
            eGravThisParticle = 6.67e-8 * self.mEnc[i-1] * self.mass[sortByRadius[i]] / self.rCentre[sortByRadius[i]]
            self.eGravTotal += eGravThisParticle
            self.eGrav[i] = self.eGrav[i-1] + eGravThisParticle
            self.mEnc[i] = self.mEnc[i-1] + self.mass[sortByRadius[i]]
            self.eTherm[i] = self.eTherm[i-1] + (3/2) * 1.38e-16 * self.temp[i] * (self.mass[iInit] / (2.38 * 1.66e-24))
            self.eKin[i] = self.eKin[i-1] + (1/2) * self.mass[i] * ((self.vx[i]-self.vx[iInit])**2 + (self.vy[i]-self.vy[iInit])**2 + (self.vz[i] - self.vz[iInit])**2)

    def temperatureDensityProfile(self, binNum=50):
        # Log density and work out bins
        numDense = np.log10(self.rho/(2.38*1.66e-24))
        densityBins = np.linspace(np.min(numDense), np.max(numDense), binNum)

        # Arrays to store values
        gasTemp = np.zeros(binNum-1)
        densityMid = np.zeros(binNum-1)
        gasStd = np.zeros(binNum-1)

        # Loop through bins and average
        for i in range(binNum-1):
            # Getting our bin ranges
            binMin = densityBins[i]
            binMax = densityBins[i+1]

            # Finding gas and temperture particles in this bin
            ind = np.where((numDense <= binMax) & (numDense >= binMin))    

            # Assigning avearage gas temperature
            gasTemp[i] = np.average(np.log10(self.temp[ind]), weights=self.mass[ind])
            gasStd[i] = np.std(np.log10(self.temp[ind]))

            densityMid[i] = (binMax + binMin) / 2

        return densityMid, gasTemp, gasStd

    # Function to determine rolling average
    def rollMean(self, data, window=10):
        rolling = np.zeros_like(data)
        for i in range(1, len(rolling)):
            if i <= window:
                rolling[i] = sum(data[0:i]) / i
            elif i > (len(rolling)-window):
                rolling[i] = sum(data[i:]) / (window-(i-len(rolling)))
            else:
                rolling[i] = sum(data[i-window:i+window]) / (2*window)

        return rolling

    # Function to plot useful information in a large panel\
    def plotPanel(self):
        # Do profiles if not already done
        self.energyProfiles()
        self.profiles(nShells=300)

        # Setup figure
        fig, axs = plt.subplots(3,3, figsize=(16,16))
 
        # RADIUS - DENSITY
        axs[0,0].loglog(self.radius[self.shellDensity!=0]/c.AU(), self.shellDensity[self.shellDensity!=0], linewidth=2, color="b")
        axs[0,0].set_xlabel("Radius [AU]")
        axs[0,0].set_ylabel("Density [$\\rm gcm^{-3}$]")

        # RADIUS - OPTICAL DEPTH
        axs[1,0].loglog(self.radius/c.AU(), self.shellOpticalDepthOut, linewidth=2, color="b")
        axs[1,0].set_xlabel("Radius [AU]")
        axs[1,0].set_ylabel("Optical Depth")

        # RADIUS - RADIAL VELOCITY
        axs[0,1].plot(self.radius/c.AU(), self.shellRadialVelocity/1e5, linewidth=2, color="b", alpha=0.1)
        axs[0,1].plot(self.radius/c.AU(), self.rollMean(self.shellRadialVelocity)/1e5, linewidth=2, color="b")
        axs[0,1].hlines(0, 1.5*np.min(self.radius/c.AU()), np.max(self.radius/c.AU()), linestyle="--", color="k")
        axs[0,1].set_xscale("log")
        axs[0,1].set_xlabel("Radius [AU]")
        axs[0,1].set_ylabel("Radial Velocity [$\\rm kms^{-1}$]")

        # RADIUS - MACH
        axs[1,1].plot(self.radius/c.AU(), self.shellRadialMach, linewidth=2, color="b", alpha=0.1)
        axs[1,1].plot(self.radius[10:]/c.AU(), self.rollMean(self.shellRadialMach)[10:], linewidth=2, color="b")
        axs[1,1].hlines(0, 1.5*np.min(self.radius/c.AU()), np.max(self.radius/c.AU()), linestyle="--", color="k")
        axs[1,1].set_xscale("log")
        axs[1,1].set_xlabel("Radius [AU]")
        axs[1,1].set_ylabel("Mach Number")

        # RADIUS - ROTATION
        axs[0,2].plot(self.radius/c.AU(), self.shellRotationalVelocity/1e5, linewidth=2, color="b", alpha=0.1)
        axs[0,2].plot(self.radius[10:]/c.AU(), self.rollMean(self.shellRotationalVelocity)[10:]/1e5, linewidth=2, color="b")
        axs[0,2].hlines(0, 1.5*np.min(self.radius/c.AU()), np.max(self.radius/c.AU()), linestyle="--", color="k")
        axs[0,2].set_xscale("log")
        axs[0,2].set_xlabel("Radius [AU]")
        axs[0,2].set_ylabel("Rotational Velocity [$\\rm kms^{-1}$]")

        # RADIUS - ENCLOSED MASS
        axs[1,2].plot(self.rCentreSorted/c.AU(), self.mEnc/c.uMass(), linewidth=2, color="b")
        axs[1,2].set_xscale("log")
        axs[1,2].set_yscale("log")
        axs[1,2].set_xlabel("Radius [AU]")
        axs[1,2].set_ylabel("Enclosed Mass [$\\rm M_\odot$]")

        # TEMPERATURE - DENSITY
        p, t, _ = self.temperatureDensityProfile()
        axs[2,0].plot(p, t, linewidth=2, color="b")
        axs[2,0].set_ylabel("Temperature [K]")
        axs[2,0].set_xlabel("Number Density [$\\rm cm^{-3}$]")

        # RADIUS - ENERGY
        axs[2,1].plot(self.rCentreSorted/c.AU(), self.eGrav, linewidth=2, color="b", label="Gravity")
        axs[2,1].plot(self.rCentreSorted/c.AU(), self.eTherm + self.eKin, linewidth=2, color="r", label="Support")
        axs[2,1].plot(self.rCentreSorted/c.AU(), self.eTherm, linewidth=2, color="orange", linestyle="--", label="Thermal")
        axs[2,1].plot(self.rCentreSorted/c.AU(), self.eKin, linewidth=2, color="green", linestyle="--", label="Kinetic")
        axs[2,1].set_xscale("log")
        axs[2,1].set_xlabel("Radius [AU]")
        axs[2,1].set_ylabel("Energy [ergs]")
        axs[2,1].legend()

        # NICE PLOT 
        weightedHist, xb, yb = np.histogram2d(self.z, self.x, weights=np.log10(self.mass), bins=(500, 500))
        histNumbers, xb, yb = np.histogram2d(self.z, self.x, bins=(500, 500))
        finalHist = weightedHist/histNumbers
        finalHist = np.ma.masked_where(histNumbers < 1, finalHist)
        axs[2,2].imshow(finalHist, aspect="auto", cmap="gist_heat", origin="lower", extent=[(yb[0]-np.median(yb))/c.AU(), (yb[-1]-np.median(yb))/c.AU(), (xb[0]-np.median(xb))/c.AU(), (xb[-1]-np.median(xb))/c.AU()])
        axs[2,2].set_xlabel("x [AU]")
        axs[2,2].set_ylabel("z [AU]")
