import numpy as np
import glob
from optparse import OptionParser
import json
from pyLCIO import IOIMPL, EVENT, UTIL

#Gather all the files you want to run over.
#Comment out all but 1 fnames
#0-50 pt muons
fnames, outName = glob.glob("/data/fmeloni/DataMuC_MuColl10_v0A/reco/muonGun_pT_0_50/muonGun_pT_0_50_reco_*.slcio"), "deltaThetaPhi0NoBIBTruth"
#50-250 pt muons
#fnames, outName = glob.glob("/data/fmeloni/DataMuC_MuColl10_v0A/reco/muonGun_pT_50_250/muonGun_pT_50_250_reco_*.slcio"), "deltaThetaPhi50NoBIBTruth"
#250-1000 pt muons
#fnames, outName = glob.glob("/data/fmeloni/DataMuC_MuColl10_v0A/reco/muonGun_pT_250_1000/muonGun_pT_250_1000_reco_*.slcio"), "deltaThetaPhi250NoBIBTruth"

parser = OptionParser()
parser.add_option('-i', '--inFile', help='--inFile Output_REC.slcio',
                  type=str, default='Output_REC.slcio')
parser.add_option('-o', '--outFile', help=f'--outFile {outName}',
                  type=str, default=outName)
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
        momentum.append(0)
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
                if (system==0):
                    hash=0
                elif (system==1):
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
            momentum[-1]=particle.getMomentumVec().Pt()

#Wrapping data into a dictionary that will be exported as a json
output={
    "deltaTheta" : deltaTheta,
    "deltaPhi" : deltaPhi,
    "pt" : momentum
}

output_json = options.outFile+".json"
with open(output_json, 'w') as fp:
    json.dump(output, fp)