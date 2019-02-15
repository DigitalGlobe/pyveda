try:
    from gbdxtools.images.base import RDABaseImage
    from gbdxtools.images.drivers import RDADaskImageDriver
    from gbdxtools.rda.interface import RDA
    rda = RDA()
    has_gbdxtools = True
except BaseException:
    has_gbdxtools = False


class MLImageDriver(RDADaskImageDriver):
    __default_options__ = {
        "proj": "EPSG:4326",
        "gsd": None,
        "pansharpen": False
    }
    image_option_support = ('proj', 'gsd', 'pansharpen')


class MLImage(RDABaseImage):
    __Driver__ = MLImageDriver
    ''' Standard image to use for RDA imagery
        - acomped
        - adjusted with Histogram DRA
        - 8 bit RGB
        - pass `pansharped=True` for pansharp imagery '''

    @classmethod
    def _build_graph(cls, cat_id, pansharpen=False, **kwargs):
        assert has_gbdxtools, 'To use MLImage gbdxtools must be installed'
        if pansharpen:
            bands = "PANSHARP"
        else:
            bands = "MS"
        strip = rda.DigitalGlobeStrip(catId=cat_id, CRS=kwargs.get("proj", "EPSG:4326"), GSD=kwargs.get("gsd", ""),
                                      correctionType="ACOMP",
                                      bands=bands,
                                      fallbackToTOA=True)

        dra = rda.HistogramDRA(strip)
        rgb = rda.SmartBandSelect(dra, bandSelection="RGB")
        return rda.Format(rgb, dataType=0)

    @property
    def _rgb_bands(self):
        return [0, 1, 2]

    @property
    def cat_id(self):
        return self.__rda_id__
