try:
    from gbdxtools.images.rda_image import RDAImage
    from gbdxtools.rda.interface import RDA
    rda = RDA()
    has_gbdxtools = True
except:
    has_gbdxtools = False

class MLImage(RDAImage):
    ''' Standard image to use for RDA imagery
        - pansharpened if possible
        - acomped
        - RGB bands adjusted with Histogram DRA '''

    def __new__(cls, cat_id, pansharpen=False, **kwargs):
        assert has_gbdxtools, 'To use MLImage gbdxtools must be installed'
        if pansharpen:
            strip = rda.DigitalGlobeStrip(catId=cat_id, CRS=kwargs.get("proj","EPSG:4326"), GSD=kwargs.get("gsd",""),
                                      correctionType="ACOMP",
                                      bands="PANSHARP",
                                      fallbackToTOA=True)
        else:
            strip = rda.DigitalGlobeStrip(catId=cat_id, CRS=kwargs.get("proj","EPSG:4326"), GSD=kwargs.get("gsd",""),
                                      correctionType="ACOMP",
                                      bands="MS",
                                      fallbackToTOA=True)

        dra = rda.HistogramDRA(strip)
        rgb = rda.SmartBandSelect(dra, bandSelection="RGB")
        img = rda.Format(rgb, dataType=0)
        self = super(MLImage, cls).__new__(cls, img)
        self.cat_id = cat_id
        self.options = {}
        return self

    @property
    def _rgb_bands(self):
        return [0,1,2]

    def __setattr__(cls, attr, value):
        return super().__setattr__(attr, value)
