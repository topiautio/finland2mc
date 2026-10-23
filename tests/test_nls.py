from finland2mc import nls


def test_tile_names_match_real_files():
    # (E, N) -> the 6 x 6 km elevation tile that covers it, as named on the NLS file service
    assert nls.tile_6k(377000, 6903000) == "N4142H"
    assert nls.tile_6k(381000, 6901000) == "N4144B"
    assert nls.tile_6k(377000, 6907000) == "N4231G"
    assert nls.tile_6k(381000, 6907000) == "N4233A"
    assert nls.tile_6k(105000, 6661000) == "K2444F"


def test_topographic_sheet_halves():
    assert nls.mtk_sheet(377000, 6903000) == "N4142R"
    assert nls.mtk_sheet(381000, 6901000) == "N4144L"


def test_sheet_bounds():
    name, bounds = nls.sheet_24k(377000, 6903000)
    assert name == "N4142"
    assert bounds == (356000, 6894000, 380000, 6906000)


def test_paths_for_an_area():
    bounds = (377984, 6901984, 384016, 6908016)  # 6 km around Keuruu centre
    assert nls.dem_paths(bounds) == [
        "dem2m/2008_latest/N4/N41/N4142H.tif", "dem2m/2008_latest/N4/N41/N4144B.tif",
        "dem2m/2008_latest/N4/N42/N4231G.tif", "dem2m/2008_latest/N4/N42/N4233A.tif"]
    assert nls.mtk_paths(bounds) == [
        "maastotietokanta/2025/shp/N4/N41/N4142R.shp.zip", "maastotietokanta/2025/shp/N4/N41/N4144L.shp.zip",
        "maastotietokanta/2025/shp/N4/N42/N4231R.shp.zip", "maastotietokanta/2025/shp/N4/N42/N4233L.shp.zip"]
