import numpy as np
import glob
from optparse import OptionParser
import json
from pyLCIO import IOIMPL, EVENT, UTIL



#First part of the code finds the cuts for the delta phi and delta theta distributions using noBIB data




#Gather all the files you want to run over.
#0-50 pt muons will be used for the lose fit because they have a realistic beamspot
fnames= glob.glob("/data/fmeloni/DataMuC_MuColl10_v0A/reco/muonGun_pT_0_50/muonGun_pT_0_50_reco_*.slcio")

parser = OptionParser()
parser.add_option('-i', '--inFile', help='--inFile Output_REC.slcio',
                  type=str, default='Output_REC.slcio')
parser.add_option('-o', '--outFile', help='--outFile deltaPhiAndThetaLooseCutData',
                  type=str, default="deltaPhiAndThetaLoseCutData")
(options, args) = parser.parse_args()

#Creating output data storage
#For each doublet layer, the hits in the layer closer to the origin are put into the inner, and hits in the layer farther away is put into the outer
innerTheta=[]
innerPhi=[]
outerTheta=[]
outerPhi=[]
deltaTheta=[]
deltaPhi=[]
#momentum has one index per event, stores 0 if the particle has decayed, and the truth particles momentum if it has not decayed
momentum=[]

#Loop over every file
for f in fnames:
    reader = IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(f)

    #Loop over events in the file
    for event in reader:
        #Adding a list to store data for this particular event to each variable
        innerTheta.append([])
        innerPhi.append([])
        outerTheta.append([])
        outerPhi.append([])
        deltaTheta.append([])
        deltaPhi.append([])
        
        #There are 9 sets of doublet layers, 1 in the barrel, 4 in the negative z endcaps, and 4 in the positive z endcaps
        #Adding one list per layer. index 0 is the barrel doublet, 1-4 are the -z endcap doubles, and 5-8 are the +z endcap doublets
        for i in range(9):
            innerTheta[-1].append(0)
            innerPhi[-1].append(0)
            outerTheta[-1].append(0)
            outerPhi[-1].append(0)
            deltaTheta[-1].append(0)
            deltaPhi[-1].append(0)

        #Accessing the collection of all digitized hits in the barrel
        hitsCollection = event.getCollection("VBTrackerHits")
        #creating a decoder that will be used layer to trace a hit back to its system and layer
        encoding=hitsCollection.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
        decoder=UTIL.BitField64(encoding)

        #collections of truth level tracks from mc particles
        mcTracksCollections=event.getCollection('MCParticle_SiTracks_Refitted')

        for relationsObject in mcTracksCollections:

            #Get the tracks from the relations object.
            track=relationsObject.getTo()
                
            #Get the hits from the trcks
            for hit in track.getTrackerHits():
                #Decoder to get the location of the hit
                cellID = int(hit.getCellID0())
                decoder.setValue(cellID)
                layer = decoder['layer'].value()
                system=decoder["system"].value()
                side = decoder["side"].value()
                
                #finding the hash for the hit so it can be put into the list
                #Barrel endcaps
                if (system==1):
                    hash=0
                elif (system==2):
                    hash=1+layer//2+(side==1)*4

                #inner hits
                if layer%2==0:
                    innerPhi[-1][hash]=hit.getPositionVec().Phi()
                    innerTheta[-1][hash]=hit.getPositionVec().Theta()
                #outer hits
                else:
                    outerPhi[-1][hash]=hit.getPositionVec().Phi()
                    outerTheta[-1][hash]=hit.getPositionVec().Theta()

            #Calculating the difference
            for i in range(9):
                #Making sure there was 1 hit on both sides of the doublet
                if (innerPhi[-1][i] !=0) and (outerPhi[-1][i] !=0):
                    deltaPhi[-1][i]=innerPhi[-1][i]-outerPhi[-1][i]
                    deltaTheta[-1][i]=innerTheta[-1][i]-outerTheta[-1][i]

            #Getting the truth information of the particle that created the track
            particle=relationsObject.getFrom()
            momentum.append(particle.getMomentumVec().Pt())

phiCut=np.zeros(9)
thetaCut=np.zeros(9)
deltaPhiArray=np.array(deltaPhi)
deltaThetaArray=np.array(deltaTheta)
for i in range(9):
    phiCut[i]=np.percentile(deltaPhiArray[(deltaPhiArray[:,i]!=0) & (np.array(momentum) >=1),i], 99.7)
    thetaCut[i]=np.percentile(deltaThetaArray[(deltaThetaArray[:,i]!=0) & (np.array(momentum) >=1),i], 99.7)



### Second part of the code uses these cuts to find their effects on the BIB data



#Creating output data storage
phiDistribution=[]
thetaDistribution=[]
momentumBIB=[]
hitSurvivalPhi=np.zeros(9)
hitSurvivalTheta=np.zeros(9)
hitSurvivalTotal=np.zeros(9)
for i in range(9):
    phiDistribution.append([])
    thetaDistribution.append([])

#Gathering all the BIB files for which we will analyze the effects of the cuts
fnamesBIB= glob.glob("/data/fmeloni/DataMuC_MuColl10_v0A/recoBIB/muonGun_pT_0_50/muonGun_pT_0_50_reco_*.slcio")

#Looping over all the BIB files
for f in fnamesBIB:
    reader = IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(f)

    #Loop over events in the file
    for event in reader:
        
        #Accessing the collection of all digitized hits in the barrel
        VBHitsCollection = event.getCollection("VBTrackerHits")

        #Accessing the collection of all digitized hits in the endcaps
        VEHitsCollection = event.getCollection("VETrackerHits")

        #creating a decoder that will be used layer to trace a hit back to its system and layer
        encoding=VBHitsCollection.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
        decoder=UTIL.BitField64(encoding)

        #collections of truth level tracks from mc particles
        mcTracksCollections=event.getCollection('MCParticle_SiTracks_Refitted')

        #Requiring that the particle has a momentum greater than 1 GeV
        particle=([relationsObject for relationsObject in mcTracksCollections])
        if len(particle)!=0:
            particle=particle[0].getFrom()
            if particle.getMomentumVec().Pt() < 1:
                continue
        else:
            continue
        
        #Looping over the objects in the collection of truth level tracks
        for relationsObject in mcTracksCollections:
            
            #Get the tracks from the relations object.
            track=relationsObject.getTo()

            #Creating a list to store the truth level phi and theta of the hits.
            #Hash is the same as before but even if the hit is on the inner layer and odd if its on the outer layer of a doublet
            truthPhi=np.zeros(18)
            truthTheta=np.zeros(18)

            #Get the hits from the trcks
            for hit in track.getTrackerHits():
                #Decoder to get the location of the hit
                cellID = int(hit.getCellID0())
                decoder.setValue(cellID)
                layer = decoder['layer'].value()
                system=decoder["system"].value()
                side = decoder["side"].value()

                #finding the hash for the hit so it can be put into the list
                #Barrel endcaps
                if (system==1):
                    hash=0
                elif (system==2):
                    hash=1+layer//2+(side==1)*4
                
                truthPhi[hash*2+layer%2]=hit.getPositionVec().Phi()
                truthTheta[hash*2+layer%2]=hit.getPositionVec().Theta()

            for num, hit in enumerate(VBHitsCollection):
                #Decoder to get the location of the hit
                cellID = int(hit.getCellID0())
                decoder.setValue(cellID)
                layer = decoder['layer'].value()

                if layer==0:
                    if np.abs(hit.getPositionVec().Phi()-truthPhi[1])<phiCut[0]:
                        phiDistribution[0].append(hit.getPositionVec().Phi()-truthPhi[1])
                        hitSurvivalPhi[0]+=1
                        if np.abs(hit.getPositionVec().Theta()-truthTheta[1])<thetaCut[0]:
                            hitSurvivalTotal[0]+=1

                    if np.abs(hit.getPositionVec().Theta()-truthTheta[1])<thetaCut[0]:
                        thetaDistribution[0].append(hit.getPositionVec().Theta()-truthTheta[1])
                        hitSurvivalTheta[0]+=1
            
            hitSurvivalTotal[0]=hitSurvivalTotal[0]/(num+1)
            hitSurvivalPhi[0]=hitSurvivalPhi[0]/(num+1)
            hitSurvivalTheta[0]=hitSurvivalTheta[0]/(num+1)

            for num, hit in enumerate(VEHitsCollection):
                #Decoder to get the location of the hit
                cellID = int(hit.getCellID0())
                decoder.setValue(cellID)
                layer = decoder['layer'].value()
                side = decoder["side"].value()

                if layer % 2 == 0:
                    if np.abs(hit.getPositionVec().Phi()-truthPhi[1+layer+(side==1)*8])<phiCut[1+layer//2+(side==1)*4]:
                        phiDistribution[1+layer//2+(side==1)*4].append(hit.getPositionVec().Phi()-truthPhi[1+layer+(side==1)*8])
                        hitSurvivalPhi[1+layer//2+(side==1)*4]+=1
                        if np.abs(hit.getPositionVec().Theta()-truthTheta[1+layer+(side==1)*8])<thetaCut[1+layer//2+(side==1)*4]:
                            hitSurvivalTotal[1+layer//2+(side==1)*4]+=1

                    if np.abs(hit.getPositionVec().Theta()-truthTheta[1+layer+(side==1)*8])<thetaCut[1+layer//2+(side==1)*4]:
                        thetaDistribution[1+layer//2+(side==1)*4].append(hit.getPositionVec().Theta()-truthTheta[1+layer+(side==1)*8])
                        hitSurvivalTheta[1+layer//2+(side==1)*4]+=1

            hitSurvivalTotal[1+layer//2+(side==1)*4]=hitSurvivalTotal[1+layer//2+(side==1)*4]/(num+1)
            hitSurvivalPhi[1+layer//2+(side==1)*4]=hitSurvivalPhi[1+layer//2+(side==1)*4]/(num+1)
            hitSurvivalTheta[1+layer//2+(side==1)*4]=hitSurvivalTheta[1+layer//2+(side==1)*4]/(num+1)
            #Getting the truth information of the particle that created the track
            particle=relationsObject.getFrom()
            momentumBIB.append(particle.getMomentumVec().Pt())


#Wrapping data into a dictionary that will be exported as a json
output={
    "truthDeltaTheta" : deltaTheta,
    "truthDeltaPhi" : deltaPhi,
    "truthPt" : momentum,
    "hitSurvivalPhi" : hitSurvivalPhi,
    "hitSurvivalTheta" : hitSurvivalTheta,
    "hitSurvivalTotal" : hitSurvivalTotal,
    "phiCut" : phiCut,
    "thetaCut" : thetaCut,
    "phiDistribution" : phiDistribution,
    "thetaDistribution" : thetaDistribution,
    "BIBPt" : momentumBIB
}

output_json = options.outFile+".json"
with open(output_json, 'w') as fp:
    json.dump(output, fp)