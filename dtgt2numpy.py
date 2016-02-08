import numpy as np
import struct as st
import sys

class dtgtReader:

    def __init__(self, tfile, hfmt=None):
        if not hfmt:
            hfmt = self.initHeader()
        self._hfmt = hfmt        
        self._shfmt = "=" + ''.join(hfmt['values']) #"=13cIIBIB" 
        self._tf = tfile
        self._dataPos = st.calcsize(self._shfmt)
        
        print 'fmt is ' + self._shfmt        
        print 'calcsize(fmt) is ' + str(st.calcsize(self._shfmt))        
        print 'tfile is ' + str(tfile)
        self.readHeader()
        
    def initHeader(self):
        # {'dat parameter name': 'struct format'}
        # numbers preside underscores for sorting and will be abandoned later
        headerFormat = {'0_cap': '13c', '1_numberOfSentences': 'I', 
                        '2_numberOfFrames': 'I', '3_isPacked': 'B',
                        '4_vectorDimension': 'I', '5_elementSize': 'B',
                        '6_hash': '16c'}
        k = headerFormat.keys()
        k.sort()
        v = [headerFormat[kv] for kv in k]
        # abandon numbers and underscores
        k = [s[s.index('_')+1:] for s in k]

        return {'names': k, 'values': v} 
    
    def readHeader(self, fname=None):
        if not fname:
            fname = self._tf
        assert fname, '_tf is not known' 
        headTup = self.unpackFromFile(self._shfmt, 0, fname)      
        self._header = self.formatHeader(headTup)          
        
        vectDim = 1 if self._header['isPacked'] else self._header['vectorDimension']   
        self._vectDim = vectDim
        self._vectSize = self._header['elementSize'] * vectDim
        self._types_struct = {1: 'B', 2: 'H', 4: 'f'}
        print self._header
        return self._header    
    
    def findOffset(self, vectNum, batchSize, batchPos, sentPos):
        needVects = batchSize - batchPos
        remVects = vectNum - sentPos
        readVects = min(needVects, remVects)
        return readVects, batchPos + readVects, sentPos+ readVects

    def readData(self, batchSize=1, batchQuantity='all'):
        """ 
        Data generator. Generates up to batchQuantity batches of size batchSize
        Output is a flat numpy array of size batchSize * self._vectDim (may be smaller for the last batch)
        """    
        assert hasattr(self, '_header'), 'run readHeader first'
        frameQuantity = batchQuantity * batchSize * self._vectDim
        if batchQuantity == 'all':
            frameQuantity = self._header['numberOfFrames']
        frameQuantity = min(frameQuantity, self._header['numberOfFrames'])
        sentPos = vectNum = frameFetched = 0 
        offset = self._dataPos
        with open(self._tf, 'rb') as f:
            f.seek(offset)
            while frameFetched < frameQuantity:
                batchSize = min(batchSize, frameQuantity - frameFetched)
                batch = np.array([])
                batchPos = 0
                while batchPos < batchSize:
                    # if wa are at the end of a sentence we should read id and vectsNum
                    if sentPos == vectNum:
                        sentPos = 0
                        (sentId, vectNum) = st.unpack('=II', f.read(st.calcsize('=II')))
                    (readVects, batchPos, sentPos)  = self.findOffset(vectNum, batchSize, batchPos, sentPos)
                    # print (vectNum, readVects, batchPos, sentPos)
                    batch = self.unpackDataFromBufferToNpA(f, batch, readVects)
                frameFetched += len(batch) / self._vectDim
                # print frameFetched, frameQuantity
                yield batch
    
    def formatHeader(self, tup):
        o = 0  
        r = {}      
        for i in range(0, len(self._hfmt['names'])):
            n = self._hfmt['names'][i]
            v = self._hfmt['values'][i]            
            fsize = filter(type(v).isdigit, v)
            if not fsize:
                fsize = 1
            fsize = int(fsize)
            t = tup[o:o+fsize]
            if fsize>1:
                r[n] = ''.join(t)
            else:
                r[n] = t[0]         
            o += fsize
        return r

    def unpackFromFile(self, fmt, offset, fname):        
        with open(fname, 'rb') as f:
            f.seek(offset)
            r = st.unpack_from(fmt, f.read(st.calcsize(fmt)))
        return r
    
    def unpackDataFromBufferToNpA(self, f, batch, readVects):
        fmt = str(readVects * self._vectDim) + self._types_struct[self._header['elementSize']]
        readBytes = readVects * self._vectSize
        # return np.append(batch, np.array(st.unpack_from(fmt, f.read(readBytes))))
        return np.concatenate((batch, np.array(st.unpack_from(fmt, f.read(readBytes)))))
        # using np.fromfile takes 3.75 ms while struct only 1.9 ms per loop
        # print np.fromfile(f, dtype=types_struct[self._header['elementSize']], count=readVects * self._vectDim)        
 
        
        

    



