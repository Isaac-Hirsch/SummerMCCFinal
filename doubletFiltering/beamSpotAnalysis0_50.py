#Imports are meant to be run on snowmass21 server
import numpy as np
import glob
from optparse import OptionParser
import json
from pyLCIO import IOIMPL, EVENT, UTIL

parser = OptionParser()
parser.add_option('-i', '--inFile', help='--inFile Output_REC.slcio',
                  type=str, default='Output_REC.slcio')
parser.add_option('-o', '--outFile', help='--outFile beamspotAnalysis0NoBIB',
                  type=str, default='beamspotAnalysis0NoBIB')
(options, args) = parser.parse_args()

#0-50 pt muons
fnames = glob.glob("/data/fmeloni/DataMuC_MuColl10_v0A/reco/muonGun_pT_0_50/muonGun_pT_0_50_reco_*.slcio")

z_0=[]
r_0=[]

#Loop over every file
for f in fnames:
    reader = IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(f)

    #Loop over events in the file
    for event in reader:
        MCParticles=event.getCollection("MCParticle")

        for particle in MCParticles:
            if particle.getGeneratorStatus()==1:
                vertex=particle.getVertexVec()
                z_0.append(vertex.Z())
                r_0.append(np.sqrt(vertex.X()**2+vertex.Y()**2))

#Wrapping data into a dictionary that will be exported as a json
output={
    "z_0": z_0,
    "r_0" : r_0,
}

output_json = options.outFile+".json"
with open(output_json, 'w') as fp:
    json.dump(output, fp)