from gbdxtools.images.rda_image import RDAImage
from gbdxtools.rda.interface import RDA

rda = RDA()

class MLImage(RDAImage):
    ''' Standard image to use for RDA imagery
        - pansharpened if possible
        - acomped
        - RGB bands adjusted with Histogram DRA '''
    
    def __new__(cls, cat_id, bands="MS", **kwargs): 
        try:
            strip = rda.DigitalGlobeStrip(catId=cat_id, CRS=kwargs.get("proj","EPSG:4326"), GSD=kwargs.get("gsd",""), 
                                      correctionType="ACOMP", 
                                      bands="PANSHARP", 
                                      fallbackToTOA=True)
        except:
            strip = rda.DigitalGlobeStrip(catId=cat_id, CRS=kwargs.get("proj","EPSG:4326"), GSD=kwargs.get("gsd",""), 
                                      correctionType="ACOMP", 
                                      bands="MS", 
                                      fallbackToTOA=True)  
        dra = rda.HistogramDRA(strip)
        rgb = rda.SmartBandSelect(dra, bandSelection="RGB")
        self = super(MLImage, cls).__new__(cls, rgb)
        self.cat_id = cat_id
        return self
    
    @property
    def _rgb_bands(self):
        return [0,1,2]