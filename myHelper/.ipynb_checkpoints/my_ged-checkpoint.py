import gmatch4py as gm

def ged_distance(g1_index, g2_index, **kwargs):
    # all edit costs are equal to 1
    ged=gm.GraphEditDistance(1,1,1,1)
    g1 = kwargs["graph_stream"][int(g1_index)]
    g2 = kwargs["graph_stream"][int(g2_index)]
    result=ged.compare([g1,g2],None)
    result = (result[0,1]+result[1,0])/2
    return result