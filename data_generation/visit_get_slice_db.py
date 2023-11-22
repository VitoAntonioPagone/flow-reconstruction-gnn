import numpy as np

# ---------------------------------------------------------
databasesPath = "/p/scratch/dems/bdanciu/nekRS/TUD/1500rpm/cyc08/grid1_600CAD/tud.nek5000"

timeStartDatabase = 4

# Output directory
outputPath = "/p/scratch/dems/bdanciu/ML/2D_slices_new/train"

# Cycle number
cycle = "cyc08"

# Reference dimensions
xref = 86  # [mm]

# ---------------------------------------------------------
OpenDatabase(databasesPath, timeStartDatabase)

AddPlot("Pseudocolor", "velocity_magnitude")
AddOperator("Slice", 1)
DrawPlots()

for state in range(timeStartDatabase, TimeSliderGetNStates(), 5):
    # Iterate through CADs
    TimeSliderSetState(state)

    Query('Time')
    t = GetQueryOutputValue()
    t = t / 9.0897044574E-03
    cad = round(t + 600)

    y_slice = np.arange(38, -39, -4)
    y_len = len(y_slice)
    y_slice = np.divide(y_slice, xref)
    if cad < 615:
        z_range = np.arange(-2.6, -63.6, -20)
    elif 615 <= cad < 642:
        z_range = np.arange(-2.6, -43.6, -20)
    elif 642 <= cad < 668:
        z_range = np.arange(-2.6, -23.6, -20)
    z_len = len(z_range) - 1
    z_range = np.divide(z_range, xref)

    for i in range(y_len):
        if 34.64 / xref < abs(y_slice[i]) < 38.72 / xref:
            x_range = np.arange(-10, 11, 20)
        elif 26.45 / xref < abs(y_slice[i]) < 34.64 / xref:
            x_range = np.arange(-20, 21, 20)
        elif 0 < abs(y_slice[i]) < 26.45 / xref:
            x_range = np.arange(-30, 31, 20)
        elif abs(y_slice[i]) == 0:
            x_range = np.arange(-40, 41, 20)
        x_len = len(x_range) - 1
        x_range = np.divide(x_range, xref)

        # Change y slice
        SetActivePlots(0)
        RemoveOperator(1, 1)
        RemoveOperator(0, 1)
        AddOperator("Slice", 1)
        SliceAtts = SliceAttributes()
        SliceAtts.originType = SliceAtts.Intercept  # Point, Intercept, Percent, Zone, Node
        SliceAtts.originPoint = (0, 0, 0)
        SliceAtts.originIntercept = y_slice[i]
        SliceAtts.originPercent = 0
        SliceAtts.originZone = 0
        SliceAtts.originNode = 0
        SliceAtts.normal = (0, -1, 0)
        SliceAtts.axisType = SliceAtts.YAxis  # XAxis, YAxis, ZAxis, Arbitrary, ThetaPhi
        SliceAtts.upAxis = (0, 0, 1)
        SliceAtts.project2d = 1
        SliceAtts.interactive = 1
        SliceAtts.flip = 0
        SliceAtts.originZoneDomain = 0
        SliceAtts.originNodeDomain = 0
        SliceAtts.meshName = "mesh"
        SliceAtts.theta = 180
        SliceAtts.phi = 0
        SetOperatorOptions(SliceAtts, -1, 1)

        for j in range(z_len):
            for k in range(x_len):
                SetActivePlots(0)
                RemoveOperator(1, 1)
                AddOperator("Box", 1)
                BoxAtts = BoxAttributes()
                BoxAtts.amount = BoxAtts.Some  # Some, All
                BoxAtts.minx = x_range[k]
                BoxAtts.maxx = x_range[k + 1]
                BoxAtts.miny = y_slice[i]
                BoxAtts.maxy = y_slice[i]
                BoxAtts.minz = z_range[j + 1]
                BoxAtts.maxz = z_range[j]
                BoxAtts.inverse = 0
                SetOperatorOptions(BoxAtts, 0, 1)
                DrawPlots()

                # Set the export database attributes.
                e = ExportDBAttributes()
                e.db_type = "VTK"
                e.variables = ("x_velocity", "y_velocity", "z_velocity", "pressure", "temperature")
                e.filename = cycle + "_" + "CAD" + str(cad) + "_Y" + str(i) + "_Z" + \
                             str(j) + "_X" + str(k)
                # Set the export directory
                e.dirname = outputPath
                opts = GetExportOptions("VTK")
                opts['FileFormat'] = "XML Binary"
                ExportDatabase(e, opts)

