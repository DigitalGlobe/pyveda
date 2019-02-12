try:
    from gbdxtools.images.base import RDABaseImage
    from gbdxtools.images.drivers import RDADaskImageDriver
    from gbdxtools.rda.interface import RDA
    rda = RDA()
    has_gbdxtools = True
except:
    has_gbdxtools = False

class MLImageDriver(RDADaskImageDriver):
    __default_options__ = {
        "proj": "EPSG:4326",
        "gsd": None
        }
    image_option_support = ('proj', 'gsd')


class MLImage(RDABaseImage):
    __Driver__ = MLImageDriver
    ''' Standard image to use for RDA imagery
        - pansharpened if possible
        - acomped
        - RGB bands adjusted with Histogram DRA '''

    @classmethod
    def _build_graph(cls, cat_id, PANSHARPEN=False, **kwargs):
        assert has_gbdxtools, 'To use MLImage gbdxtools must be installed'
        if 'PANSHARPEN': 
            bands = "PANSHARP"
        else:
            bands = "MS"
        strip = rda.DigitalGlobeStrip(catId=cat_id, CRS=kwargs.get("proj","EPSG:4326"), GSD=kwargs.get("gsd",""),
                                      correctionType="ACOMP",
                                      bands=bands,
                                      fallbackToTOA=True)

        dra = rda.HistogramDRA(strip)
        rgb = rda.SmartBandSelect(dra, bandSelection="RGB")
        return rda.Format(rgb, dataType=0)

    @property
    def _rgb_bands(self):
        return [0,1,2]

    @property
    def cat_id(self):
        return self.__rda_id__


