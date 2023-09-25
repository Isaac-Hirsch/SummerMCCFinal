import numpy as np
import glob
from optparse import OptionParser
import json
from pyLCIO import IOIMPL, EVENT, UTIL

#Gather all the files you want to run over.
#Comment out all but 1 fnames
#BIB
#fnames, usingTruth, outName = glob.glob("/data/fmeloni/DataMuC_MuColl10_v0A/recoBIB/photonGun_pT_0_50/photonGun_pT_0_50_reco_2000.slcio"), False, "nearestPair0BIB"
#0-50 pt muons
fnames, usingTruth, outName = glob.glob("/data/fmeloni/DataMuC_MuColl10_v0A/reco/muonGun_pT_0_50/muonGun_pT_0_50_reco_*.slcio"), True, "nearestPair0NoBIB"
#250-1000 pt muons
#fnames, usingTruth, outName = glob.glob("/data/fmeloni/DataMuC_MuColl10_v0A/reco/muonGun_pT_250_1000/muonGun_pT_250_1000_reco_*.slcio"), True, "nearestPair250NoBIB"

parser = OptionParser()
parser.add_option('-i', '--inFile', help='--inFile Output_REC.slcio',
                  type=str, default='Output_REC.slcio')
parser.add_option('-o', '--outFile', help=f'--outFile {outName}',
                  type=str, default=outName)
(options, args) = parser.parse_args()

#Setting number of bins for the sorting function
#Pseudorapditity spans -2.4 to 2.4
nPseudoRap=100
startPseudo=-2.4
endPseudo=2.4
#Phi spans -pi to pi
nPhi=100
startPhi=-np.pi
endPhi=np.pi

class breadthFirstSearch(object):
    #Object for breadth-first search to find the nearest hit in boxes to a point.
    def __init__(self, boxes: list, points: list):
        #Boxes should be a list of lists of hits on the second doublet layer where the first list represents the pseudo hash and the second represents the phi hash
        #Points should be a list of 2 length tuples that represent all hits on the first doublet layer. The first element of the tuple should be pseudorapidity and the second should be phi
        #Asserting boxes and points are the correct format
        assert type(boxes)==list, "box is wrong type. It is " + str(type(boxes))
        for box in boxes:
            assert type(box)==list, "an element of box is the wrong type. It is " + str(type(box))
        assert type(points)==list, "points is wrong type. It is " + str(type(points))
        for point in points:
            assert type(point)==tuple, "an element of points is the wrong type. It is " + str(type(point))
            assert len(point)==2, "an element of points contains too many elements. It has " + str(len(point)) + " elements."
        self.boxes=boxes
        self.points=points

    def search(self)-> list:
        #Searches for the nearest hit in box for all points, returns a list of tuples (delta R,delta prseudorapditity, delta phi)
        #Ouput stores the final delta R values in the first list, the final delta pseudorapidity in the second list, and the final delta phi in the last list
        output=[[],[],[]]
        #Loop over all hits on the first doublet layer to find each of their closest points
        for pseudo, phi in self.points:
            assert type(pseudo)==float, "the first element in a point is the wrong type. It is a " + str(type(pseudo))
            assert type(phi)==float, "the second element in a point is the wrong type. It is a " + str(type(phi))
            #Starting a priority queue for breadth first search and intializing the current box the hit is in as the first seach
            #All of the queues are synced so that indexing the nth index of all of them returns info on the same box
            #If this version is too slow or takes up too much memory, you can change it such that the search removes items from the queue after it looks at them
            pseudoIndex=int(((2.4+pseudo)*nPseudoRap)/4.8)
            phiIndex=int(((np.pi+phi)*nPhi)/(2*np.pi))
            boxQueue=[self.boxes[pseudoIndex][phiIndex]]
            RQueue=[0]
            #Keep track of the steps to get to each box from the point
            #Specifically if phi steps=0 then the closest point is on the wall of the box with the same phi value
            #If phi steps<0, then the closest point is on the wall with the maximum phi value
            #If phi steps>0, then the closest point is on the wall with the minimum phi value
            #The above statements are also true for pseudorapidity
            phiStepsQueue=[0]
            pseudoStepsQue=[0]
            #repeatDict is a dictionary of all the steps already searched and will be used to remember if we have already searched a box
            #keys will be tuples (pseudoSteps, phiSteps)
            repeatDict={(0,0) : True}
            #Searching through priority queue until we can guarentee we have found that closest point
            i=0
            minR=float('inf')
            minPhi=float('inf')
            minPseudo=float('inf')
            #While loop will stop as soon as the box with the closest possible point is farther then the closests point we already found
            while RQueue[i] < minR:
                #Look through the next element in the queue for the closest hit to our point
                boxResult=self.searchBox(boxQueue[i], pseudo, phi)
                #If there is at least 1 hit in the box, check if its the closest we have found so far
                if boxResult:
                    if boxResult[0] < minR:
                        minR=boxResult[0]
                        minPseudo=boxResult[1]
                        minPhi=boxResult[2]
                #Add appropriate neighboring boxes to the priority queue
                for pseudoStep, phiStep in ((-1,0),(1,0),(0,-1),(0,1)):
                    #calculate total steps from the original box the point would have been in
                    totPhiSteps=phiStepsQueue[i]+phiStep
                    totPseudoSteps=pseudoStepsQue[i]+pseudoStep
                    #require the next box to be one we have not seen before and to within out psuedo and phi bounds
                    if (not ((totPseudoSteps,totPhiSteps) in repeatDict)) & (np.abs(pseudo+totPseudoSteps*4.8/nPseudoRap) < 2.4) & (np.abs(phi + totPhiSteps*2*np.pi/nPhi) <= np.pi):
                        #Add the box to the list of boxes we have already seen
                        repeatDict[(totPseudoSteps, totPhiSteps)]=True
                        #Calculate the nearest possible R
                        if totPseudoSteps==0:
                            deltaMinPseudo=0
                        elif totPseudoSteps<0:
                            deltaMinPseudo= -((2.4+pseudo)%(4.8/nPseudoRap))+(totPseudoSteps+1)*4.8/nPseudoRap
                        else:
                            deltaMinPseudo= -((2.4+pseudo)%(4.8/nPseudoRap))+totPseudoSteps*4.8/nPseudoRap
                        if totPhiSteps==0:
                            deltaMinPhi=0
                        elif totPhiSteps<0:
                            deltaMinPhi= -((np.pi+phi)%(2*np.pi/nPhi))+(totPhiSteps+1)*2*np.pi/nPhi
                        else:
                            deltaMinPhi= -((np.pi+phi)%(2*np.pi/nPhi))+totPhiSteps*2*np.pi/nPhi
                        deltaMinR=np.sqrt(deltaMinPhi**2+deltaMinPseudo**2)
                        #check whether the box might contain the closest point
                        if deltaMinR < minR:
                            #Find where to insert it in sorted RQueue array and then assert all the other values in the same location in the other arrays
                            RLen=len(RQueue)
                            for j in range(RLen-i):
                                if RQueue[-j] < deltaMinR:
                                    #Must of RLen-j instead of just -j, because -j inserts the item before the index while RLen-j inserts it after the index
                                    RQueue.insert(RLen-j, deltaMinR)
                                    boxQueue.insert(RLen-j,self.boxes[pseudoIndex + totPseudoSteps][phiIndex+totPhiSteps])
                                    phiStepsQueue.insert(RLen-j,totPhiSteps)
                                    pseudoStepsQue.insert(RLen-j,totPseudoSteps)
                                    break
                i+=1
                if len(RQueue)==i:
                    break
            output[0].append(minR)  
            output[1].append(minPseudo)
            output[2].append(minPhi)
        return output      

    def searchBox(self, box: list, pseudo: float,phi: float):
        #Searches a box for the nearest hit in delta R to (pseudo, phi)
        #If the box is empty it returns False, otherwise it returns (minR,minPseudo,minPhi) of the closest hit
        if len(box):
            assert len(box[0])==2, "a box has a hit of the wrong size. It is size " + str(len(box[0]))
            assert type(box[0][0])==float, "the first element of a hit in a box is the wrong type. It is a " + str(type(box[0][0]))
            assert type(box[0][1])==float, "the second element of a hit in a box is the wrong type. It is a " + str(type(box[0][1]))
            minPseudo=np.abs(box[0][0]-pseudo)
            minPhi=np.abs(box[0][1]-phi)
            minR=np.sqrt(minPseudo**2+minPhi**2)
            for boxPseudo, boxPhi in box[1:]:
                assert type(boxPseudo)==float, "the first element of a hit in a box is the wrong type. It is a " + str(type(box[0][0]))
                assert type(boxPhi)==float, "the second element of a hit in a box is the wrong type. It is a " + str(type(box[0][1]))
                deltaPseudo=np.abs(boxPseudo-pseudo)
                deltaPhi=np.abs(boxPhi-phi)
                deltaR=np.sqrt(deltaPhi**2+deltaPseudo**2)
                if deltaR < minR:
                    minR=deltaR
                    minPhi=deltaPhi
                    minPseudo=deltaPseudo
            return (minR,minPseudo,minPhi)
        else:
            return False





#Creating output data storage
deltaPseudo=[]
deltaPhi=[]
deltaR=[]
#nHoles stores a 0 if there is a hit (or hits) on both sides of a doublet layer, 1 if theres a hit on only 1 side of the doublet layer, and 2 if there are hits on both
nHoles=[]
#momentum has one index per event, stores 0 if the particle has decayed, and the truth particles momentum if it has not decayed
momentum=[]
#Keep track os pseudo and phi of hits
hitPseudo=[]
hitPhi=[]

#Loop over every file
for f in fnames:
    reader = IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(f)

    #Loop over events in the file
    for event in reader:
        #Adding a list to store data for this particular event to each variable
        deltaPseudo.append([])
        deltaPhi.append([])
        deltaR.append([])
        nHoles.append([])
        hitPseudo.append([])
        hitPhi.append([])
        momentum.append(0)

        for i in range(9):
            deltaPseudo[-1].append([])
            deltaPhi[-1].append([])
            deltaR[-1].append([])
            nHoles[-1].append(0)
            hitPseudo[-1].append([])
            hitPhi[-1].append([])





        #Find out if the particle has decayed and if not, what is its pt
        #Collecting all the MCParticles
        MCParticles = [particle for particle in event.getCollection("MCParticle")]
        if (len(MCParticles)==1) and (MCParticles[0].getGeneratorStatus()==1):
            momentum[-1]=MCParticles[0].getMomentumVec().Pt()




        #The below code searches over the vertex barrel

        #Accessing the collection of all digitized hits in the barrel
        hitsCollection = event.getCollection("VBTrackerHits")
        #creating a decoder that will be used layer to trace a hit back to its system and layer
        encoding=hitsCollection.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
        decoder=UTIL.BitField64(encoding)

        #Can be used to determine if a hit was created by a truth particle
        relationCollection=event.getCollection('VBTrackerHitsRelations')
        relation = UTIL.LCRelationNavigator(relationCollection)

        #Creating bins to sort hits from the second layer into.
        #The Pseudonapidity spans -2.4 to 2.4 and is split into nPsuedoRap evenly sized bins
        #The Phi spans -pi to pi and is split into nPhi evenly sized bins
        sorting=[]
        for i in range(nPseudoRap):
            sorting.append([])
            for k in range(nPhi):
                sorting[i].append([])
        #Calculating if there is at least 1 hit in both layers. If its only 1, well record a hole
        noHits=[True,True]

        firstLayerHit=[]

        #Looping over all hits in the collection
        for hit in hitsCollection:
            
            #Detcting if the hit is connected with a muon if 
            if usingTruth:
                if relation.getRelatedToObjects(hit)[0].getMCParticle().getGeneratorStatus() != 1:
                    continue
            

            #Decoder
            cellID = int(hit.getCellID0())
            decoder.setValue(cellID)
            layer = decoder['layer'].value()
            
            #This is the second layer of the only vertex barrel doublet
            if layer==1:
                #Checking if we have found at least 1 hit in this layer
                if noHits[0]:
                    noHits[0]=False
                pseudoRapidity=hit.getPositionVec().PseudoRapidity()
                phi=hit.getPositionVec().Phi()
                #Hashing the hit into the correct bins
                sorting[int(((2.4+pseudoRapidity)*nPseudoRap)/4.8)][int(((np.pi+phi)*nPhi)/(2*np.pi))].append((pseudoRapidity,phi))

            #Adding hits from the first doublet layer to a list so we can find their closest pairs later
            elif layer==0:
                #Checking if we have found at least 1 hit in this layer
                if noHits[1]:
                    noHits[1]=False
                firstLayerHit.append((hit.getPositionVec().PseudoRapidity(),hit.getPositionVec().Phi()))
                hitPseudo[-1][0].append(hit.getPositionVec().PseudoRapidity())
                hitPhi[-1][0].append(hit.getPositionVec().Phi())

        
        #Checking if their is at least 1 hit on both layers of the doublet
        if noHits[0] and  noHits[1]:
            nHoles[-1][0]=2
        elif noHits[0] or  noHits[1]:
            nHoles[-1][0]=1
        #Run through every hit in the first layer of the doublet and find its nearest neighbor in terms of delta R on the second layer using breadth-first search
        else:
            barrelSearch=breadthFirstSearch(sorting,firstLayerHit)
            deltaR[-1][0], deltaPseudo[-1][0], deltaPhi[-1][0] = barrelSearch.search()






        #Below code repeats the above process for the endcaps
                
        #Accessing data for the four doublet layer in the vertex endcaps
        hitsCollection = event.getCollection("VETrackerHits")

        #Can be used to determine if a hit was created by a truth particle
        relationCollection=event.getCollection('VETrackerHitsRelations')
        relation = UTIL.LCRelationNavigator(relationCollection)

        #Reseting sorting function and firstLayerHit
        sorting=[]
        firstLayerHit=[]
        #Calculating if there is at least 1 hit in each 2nd doublet layer. If not, then the event has a hole on that doublet layer
        noHits=[]
        #Appending a list for each endcap doublet. index 0-3 are the -z endcaps from innermost to outer most and 4-7 are the +z endcaps also starting with innermost
        for i in range(8):
            sorting.append([])
            firstLayerHit.append([])
            noHits.append([True,True])
            for j in range(nPseudoRap):
                sorting[i].append([])
                for k in range(nPhi):
                    sorting[i][j].append([])

        #Looping over all hits in the collection
        for hit in hitsCollection:

            #Detcting if the hit is connected with a muon if 
            if usingTruth:
                if relation.getRelatedToObjects(hit)[0].getMCParticle().getGeneratorStatus() != 1:
                    continue
            
            #Decoder
            cellID = int(hit.getCellID0())
            decoder.setValue(cellID)
            layer = decoder['layer'].value()
            side = decoder['side'].value()
            #layer/2+4*(side==1) uniquely hashes each doublet endcap into a value of 0-7 (but both layers from the same doublet have the same index)
            index=int(layer/2+4*(side==1))
            #Identifying any hit that is the second layer of any of the four doublets
            if (layer==1) | (layer==3) | (layer==5) | (layer==7):
                #Checking if there is at least 1 hit in this layer
                if noHits[index][0]:
                    noHits[index][0]=False
                pseudoRapidity=hit.getPositionVec().PseudoRapidity()
                phi=hit.getPositionVec().Phi()
                sorting[index][int(((2.4+pseudoRapidity)*nPseudoRap)/4.8)][int(((np.pi+phi)*nPhi)/(2*np.pi))].append((pseudoRapidity,phi))

            #All other hits are in the first layer of a doublet
            else:
                #Checking if there is at least 1 hit in this layer
                if noHits[index][1]:
                    noHits[index][1]=False
                firstLayerHit[index].append((hit.getPositionVec().PseudoRapidity(),hit.getPositionVec().Phi()))
                hitPseudo[-1][1+index].append(hit.getPositionVec().PseudoRapidity())
                hitPhi[-1][1+index].append(hit.getPositionVec().Phi())

        #Run through every hit in the first layer of each doublet and find its nearest neighbor in terms of delta R on the second layer using breadth-first search    
        for i in range(8):
            if noHits[i][0] and noHits[i][1]:
                nHoles[-1][1+i]=2
            elif noHits[i][0] or noHits[i][1]:
                nHoles[-1][1+i]=1
            else:
                endcapSearch=breadthFirstSearch(sorting[i],firstLayerHit[i])
                deltaR[-1][i+1], deltaPseudo[-1][i+1], deltaPhi[-1][i+1] = endcapSearch.search()






#Wrapping data into a dictionary that will be exported as a json
output={
    "deltaPesudorapdity" : deltaPseudo,
    "deltaPhi" : deltaPhi,
    "deltaR" : deltaR,
    "nHoles" : nHoles,
    "momentum" : momentum,
    "hitPseudo" : hitPseudo,
    "hitPhi" : hitPhi
}

output_json = options.outFile+".json"
with open(output_json, 'w') as fp:
    json.dump(output, fp)