import random
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from xml.dom.minidom import parse, parseString

pl = st.empty()


def geo_limits(geo_xml, unit):
    geometry_wall = read_subroom_walls(geo_xml, unit)
    geominX = 1000
    geomaxX = -1000
    geominY = 1000
    geomaxY = -1000
    Xmin = []
    Ymin = []
    Xmax = []
    Ymax = []
    for _, wall in geometry_wall.items():
        Xmin.append(np.min(wall[:, 0]))
        Ymin.append(np.min(wall[:, 1]))
        Xmax.append(np.max(wall[:, 0]))
        Ymax.append(np.max(wall[:, 1]))

    geominX = np.min(Xmin)
    geomaxX = np.max(Xmax)
    geominY = np.min(Ymin)
    geomaxY = np.max(Ymax)
    return geominX, geomaxX, geominY, geomaxY


def read_subroom_walls(xml_doc, unit):
    dict_polynom_wall = {}
    n_wall = 0
    if unit == "cm":
        cm2m = 100
    else:
        cm2m = 1

    for _, s_elem in enumerate(xml_doc.getElementsByTagName("subroom")):
        for _, p_elem in enumerate(s_elem.getElementsByTagName("polygon")):
            if True or p_elem.getAttribute("caption") == "wall":
                n_wall = n_wall + 1
                n_vertex = len(p_elem.getElementsByTagName("vertex"))
                vertex_array = np.zeros((n_vertex, 2))
                for v_num, _ in enumerate(p_elem.getElementsByTagName("vertex")):
                    vertex_array[v_num, 0] = (
                        p_elem.getElementsByTagName("vertex")[v_num]
                        .attributes["px"]
                        .value
                    )
                    vertex_array[v_num, 1] = (
                        p_elem.getElementsByTagName("vertex")[v_num]
                        .attributes["py"]
                        .value
                    )

                dict_polynom_wall[n_wall] = vertex_array / cm2m

    return dict_polynom_wall


def plot_trajectories(
    geo_walls,
    min_x,
    max_x,
    min_y,
    max_y,
):    
    fig = make_subplots(rows=1, cols=1, subplot_titles=["<b>Geometry</b>"])
    
    for gw in geo_walls.keys():
        trace_walls = go.Scatter(
            x=geo_walls[gw][:, 0],
            y=geo_walls[gw][:, 1],
            showlegend=False,
            mode="lines",
            line=dict(color="black", width=2),
        )
        fig.append_trace(trace_walls, row=1, col=1)

    eps = 1
    fig.update_yaxes(
        range=[min_y - eps, max_y + eps],
    )
    fig.update_xaxes(
        range=[min_x - eps, max_x + eps],
    )
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
  )
    return fig


def plot_geometry(ax, _geometry_wall):
    for gw in _geometry_wall.keys():
        ax.plot(_geometry_wall[gw][:, 0], _geometry_wall[gw][:, 1], color="white", lw=2)


def make_circle(centerX, centerY, r):
    X = []
    Y = []
    for i in np.arange(0, 100, 0.1):
        theta0 = np.pi/2 + i * np.pi/100
        x = centerX + r * np.cos(theta0)
        y = centerY + r * np.sin(theta0)
        X.append(x)
        Y.append(y)

    return X, Y

def within_geometry(x, y, geoMinX, geoMaxX, geoMinY, geoMaxY):
    return x > geominX and x < geomaxX and y > geominY and y < geomaxY
            

def generate_random(N, r1, r2, ped_r, centerX, centerY, geoMinX, geoMaxX, geoMinY, geoMaxY):
    # biggest radius
    if r1 > r2:
        rmax = r1
        rmin = r2
    else:
        rmax = r2
        rmin = r1

    possible_peds = []
    Rmax = rmax
    while rmax > rmin+ped_r:
        print("-----")
        print(rmin, rmax)
        
        rmax -= 2*ped_r
        delta_theta = 2 * ped_r / rmax
        N_possible = int(np.pi / delta_theta)
        for i in np.arange(0.5, N_possible):
            theta0 = i * delta_theta + np.pi/2
            print(f"theta {theta0}, {i}")
            x = centerX + rmax * np.cos(theta0)
            y = centerY + rmax * np.sin(theta0)
            if within_geometry(x, y, geoMinX, geoMaxX, geoMinY, geoMaxY):
                possible_peds.append((x, y))


    pl.info(f"Possible positions {len(possible_peds)}")
    if N <= len(possible_peds):
        select_peds = random.sample(possible_peds, N)  # np.pi * random
    else:
        pl.warning(f"Wanted {N} peds but only {len(possible_peds)} are possible")
        select_peds = possible_peds
        
    peds = []
    for x, y in select_peds:
        peds.append((x,y))

    return peds
    
    
if __name__ == "__main__":
    geometry_file = st.sidebar.file_uploader(
        "🏠 Geometry file ",
        type=["xml"],
        help="Load geometry file",
    )    
    if geometry_file:
        # ------ UI
        #st.sidebar.write("#### Area 1")
        c1, c2 = st.sidebar.columns((1, 1))
        rmax = c1.slider("r_max1", 1.0, 3.0, 2.0, step=0.1)
        rmin = c2.slider("r_min1", 0.1, rmax-0.5, 0.6, step=0.1)
        #st.sidebar.write("#### Area 2")
        rmax2 = c1.slider("r_max2", 1.0, 3.0, 2.0, step=0.1)
        rmin2 = rmax
        #st.sidebar.write("#### Area 3")
        rmax2 = c2.slider("r_max3", 1.0, 3.0, 2.0, step=0.1)
        rmin2 = rmax2        
        rped = st.sidebar.slider("r_ped", 0.1, 0.5, 0.1)
        st.sidebar.write("#### Origin")
        center_x = st.sidebar.number_input('Center x', value=60.0, step=0.1)
        center_y = st.sidebar.number_input('Center y', value=102.0, step=0.1)
        N = st.slider("N", 10, 50, 1)

        #-----------
        geo_xml = parseString(geometry_file.getvalue())    
        geometry_walls = read_subroom_walls(geo_xml, unit="m")
        geominX, geomaxX, geominY, geomaxY = geo_limits(
            geo_xml, unit="m"
        )
        peds = generate_random(N, rmin, rmax, rped, center_x, center_y, geominX, geomaxX, geominY, geomaxY)
        fig = plot_trajectories(
            geometry_walls,
            geominX,
            geomaxX,
            geominY,
            geomaxY)


        X, Y = make_circle(center_x, center_y, rmax)
        circle1 = go.Scatter(
            x=X,
            y=Y,
            showlegend=False,
            mode="lines",
            line=dict(color="red", width=2),
        )
        X, Y = make_circle(center_x, center_y, rmin)
        fig.append_trace(circle1, row=1, col=1)
        
        circle2 = go.Scatter(
            x=X,
            y=Y,
            showlegend=False,
            mode="lines",
            line=dict(color="blue", width=2),
        )           
        fig.append_trace(circle2, row=1, col=1)
        for ped in peds:
            x, y = ped
            fig.add_shape(type="circle",
                          xref="x", yref="y",
                          x0=x-rped,
                          y0=y-rped,
                          x1=x+rped,
                          y1=y+rped,
                          fillcolor="PaleTurquoise",
                          line_color="LightSeaGreen",
                          )


        
        st.plotly_chart(fig, use_container_width=True)
