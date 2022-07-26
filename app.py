import random
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse, parseString
from itertools import product
import numpy as np
import plotly.graph_objs as go
import streamlit as st
from plotly.subplots import make_subplots
from zipfile import ZipFile

st.set_page_config(
    page_title="JuPedSim: Make inifile",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/jupedsim/jpscore",
        "Report a bug": "https://github.com/jupedsim/jpscore/issues",
        "About": "Open source framework for simulating, analyzing and visualizing pedestrian dynamics",
    },
)

pl = st.empty()


def write_inifile(
    r1,
    v01,
    sigma_v0_1,
    T1,
    sigma_T_1,
    peds1,
    r2,
    v02,
    sigma_v0_2,
    T2,
    sigma_T_2,
    peds2,
    r3,
    v03,
    sigma_v0_3,
    T3,
    sigma_T_3,
    peds3,
    ini_file,
    geometry_file,
):
    global_group_id = 1
    # --------
    # create_geo_header
    data = ET.Element("JuPedSim")
    data.set("project", "Ezel")
    data.set("version", "0.9")
    # make room/subroom
    header = ET.SubElement(data, "header")
    seed = ET.SubElement(header, "seed")
    seed.text = "1234"
    max_sim_time = ET.SubElement(header, "max_sim_time")
    max_sim_time.text = "200"
    geometry_f = ET.SubElement(header, "geometry")
    geometry_f.text = geometry_file.name
    traj = ET.SubElement(header, "trajectories")
    traj.set("format", "plain")
    traj.set("fps", "16")
    traj.set("precision", "4")
    traj.set("color_mode", "group")
    file_l = ET.SubElement(traj, "file")
    #traj_name = f"trajectories_R1_{r1:.1f}_V1_{v01:.1f}_T1_{T1:.1f}_R2_{r2:.1f}_V2_{v02:.1f}_T2_{T2:.1f}_R3_{r3:.1f}_V3_{v03:.1f}_T3_{T3:.1f}"
    traj_name = "traj_" + ini_file.split("ini_")[-1]
    print(ini_file)
    print(traj_name)
    file_l.set("location", traj_name)
    agents = ET.SubElement(data, "agents")
    agents.set("operational_model_id", "3")
    dist = ET.SubElement(agents, "agents_distribution")
    # print("peds", peds1)
    for (X1, Y1) in peds1:
        # st.code(f"G1: {X1:.3f}, {Y1:.3f}")
        group = ET.SubElement(dist, "group")
        group.set("group_id", f"{global_group_id}")
        # global_group_id += 1
        group.set("agent_parameter_id", "1")
        group.set("room_id", "1")
        group.set("subroom_id", "0")
        group.set("number", "1")
        group.set("router_id", "1")
        group.set("startX", f"{X1:.3f}")
        group.set("startY", f"{Y1:.3f}")

    global_group_id += 1
    # -----
    # print("peds2", peds2)
    for (X2, Y2) in peds2:
        # st.code(f"G2: {X2:.3f}, {Y2:.3f}")
        group = ET.SubElement(dist, "group")
        group.set("group_id", f"{global_group_id}")
        # global_group_id += 1
        group.set("agent_parameter_id", "2")
        group.set("room_id", "1")
        group.set("subroom_id", "0")
        group.set("number", "1")
        group.set("router_id", "1")
        group.set("startX", f"{X2:.3f}")
        group.set("startY", f"{Y2:.3f}")

    # -----
    global_group_id += 1
    #    print("peds3", peds3)
    for (X3, Y3) in peds3:
        #        print(peds3)
        #       st.code(f"G3: {X3:.3f}, {Y3:.3f}")
        group = ET.SubElement(dist, "group")
        group.set("group_id", f"{global_group_id}")
        # global_group_id += 1
        group.set("agent_parameter_id", "3")
        group.set("room_id", "1")
        group.set("subroom_id", "0")
        group.set("number", "1")
        group.set("router_id", "1")
        group.set("startX", f"{X3:.3f}")
        group.set("startY", f"{Y3:.3f}")

    operational = ET.SubElement(data, "operational_models")
    model = ET.SubElement(operational, "model")
    model.set("operational_model_id", "3")
    model.set("description", "Tordeux2015")
    parameters = ET.SubElement(model, "model_parameters")
    step = ET.SubElement(parameters, "stepsize")
    step.text = "0.01"
    exit_str = ET.SubElement(parameters, "exit_crossing_strategy")
    exit_str.text = "3"
    lcells = ET.SubElement(parameters, "linkedcells")
    lcells.set("enabled", "true")
    lcells.set("cell_size", "3")
    force_ped = ET.SubElement(parameters, "force_ped")
    force_ped.set("a", "8")
    force_ped.set("D", "0.1")
    force_wall = ET.SubElement(parameters, "force_wall")
    force_wall.set("a", "5")
    force_wall.set("D", "0.02")
    # -------
    agent_parameters = ET.SubElement(model, "agent_parameters")
    agent_parameters.set("agent_parameter_id", "1")
    v0 = ET.SubElement(agent_parameters, "v0")
    v0.set("mu", f"{v01:.1f}")
    v0.set("sigma", f"{sigma_v0_1:.1f}")
    bmax = ET.SubElement(agent_parameters, "bmax")
    bmax.set("mu", f"{r1:.1f}")
    bmax.set("sigma", "0")
    bmin = ET.SubElement(agent_parameters, "bmin")
    bmin.set("mu", f"{r1:.1f}")
    bmin.set("sigma", "0")
    amin = ET.SubElement(agent_parameters, "amin")
    amin.set("mu", f"{r1:.1f}")
    amin.set("sigma", "0")
    tau = ET.SubElement(agent_parameters, "tau")
    tau.set("mu", "0.5")
    tau.set("sigma", "0")
    atau = ET.SubElement(agent_parameters, "atau")
    atau.set("mu", "0")
    atau.set("sigma", "0")
    T = ET.SubElement(agent_parameters, "T")
    T.set("mu", f"{T1:.1f}")
    T.set("sigma", "0")
    # -------
    agent_parameters = ET.SubElement(model, "agent_parameters")
    agent_parameters.set("agent_parameter_id", "2")
    v0 = ET.SubElement(agent_parameters, "v0")
    v0.set("mu", f"{v02:.1f}")
    v0.set("sigma", "0")
    bmax = ET.SubElement(agent_parameters, "bmax")
    bmax.set("mu", f"{r2:.1f}")
    bmax.set("sigma", "0")
    bmin = ET.SubElement(agent_parameters, "bmin")
    bmin.set("mu", f"{r2:.1f}")
    bmin.set("sigma", "0")
    amin = ET.SubElement(agent_parameters, "amin")
    amin.set("mu", f"{r2:.1f}")
    amin.set("sigma", "0")
    tau = ET.SubElement(agent_parameters, "tau")
    tau.set("mu", "0.5")
    tau.set("sigma", "0")
    atau = ET.SubElement(agent_parameters, "atau")
    atau.set("mu", "0")
    atau.set("sigma", "0")
    T = ET.SubElement(agent_parameters, "T")
    T.set("mu", f"{T2:.1f}")
    T.set("sigma", "0")
    # -------
    agent_parameters = ET.SubElement(model, "agent_parameters")
    agent_parameters.set("agent_parameter_id", "3")
    v0 = ET.SubElement(agent_parameters, "v0")
    v0.set("mu", f"{v03:.1f}")
    v0.set("sigma", "0")
    bmax = ET.SubElement(agent_parameters, "bmax")
    bmax.set("mu", f"{r3:.1f}")
    bmax.set("sigma", "0")
    bmin = ET.SubElement(agent_parameters, "bmin")
    bmin.set("mu", f"{r3:.1f}")
    bmin.set("sigma", "0")
    amin = ET.SubElement(agent_parameters, "amin")
    amin.set("mu", f"{r3:.1f}")
    amin.set("sigma", "0")
    tau = ET.SubElement(agent_parameters, "tau")
    tau.set("mu", "0.5")
    tau.set("sigma", "0")
    atau = ET.SubElement(agent_parameters, "atau")
    atau.set("mu", "0")
    atau.set("sigma", "0")
    T = ET.SubElement(agent_parameters, "T")
    T.set("mu", f"{T3:.1f}")
    T.set("sigma", "0")
    router_choice = ET.SubElement(data, "route_choice_models")
    router = ET.SubElement(router_choice, "router")
    router.set("router_id", "1")
    router.set("description", "global_shortest")

    b_xml = ET.tostring(data, encoding="utf8", method="xml")
    b_xml = prettify(b_xml)

    # st.code(b_xml, language="xml")
    with open(ini_file, "w") as f:
        f.write(b_xml)

    return b_xml


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    reparsed = parseString(elem)
    return reparsed.toprettyxml(indent="\t")


def geo_limits(geo_xml, unit):
    geometry_wall = read_subroom_walls(geo_xml, unit)
    #print(geometry_wall)
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
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def plot_geometry(ax, _geometry_wall):
    for gw in _geometry_wall.keys():
        ax.plot(_geometry_wall[gw][:, 0], _geometry_wall[gw][:, 1], color="white", lw=2)


def make_circle(centerX, centerY, r):
    X = []
    Y = []
    for i in np.arange(0, 100, 0.1):
        theta0 = np.pi / 2 + i * np.pi / 100
        x = centerX + r * np.cos(theta0)
        y = centerY + r * np.sin(theta0)
        X.append(x)
        Y.append(y)

    return X, Y


def within_geometry(x, y, geoMinX, geoMaxX, geoMinY, geoMaxY):
    return x > geoMinX and x < geoMaxX and y > geoMinY and y < geoMaxY


@st.cache
def area(r1, r2, ped_r, centerX, centerY, _geoMinX, _geoMaxX, _geoMinY, _geoMaxY):
    # biggest radius
    if r1 > r2:
        rmax = r1
        rmin = r2
    else:
        rmax = r2
        rmin = r1

    possible_peds = []
    Rmax = rmax
    while rmax > rmin + ped_r + 0.2:
        rmax -= 2 * ped_r
        delta_theta = 2 * ped_r / rmax
        N_possible = int(np.pi / delta_theta)
        for i in np.arange(0.5, N_possible):
            theta0 = i * delta_theta + np.pi / 2
            # print(f"theta {theta0}, {i}")
            x = centerX + rmax * np.cos(theta0)
            y = centerY + rmax * np.sin(theta0)
            if within_geometry(x, y, _geoMinX, _geoMaxX, _geoMinY, _geoMaxY):
                possible_peds.append((x, y))

    area = len(possible_peds) * np.pi * ped_r**2

    return area


@st.cache
def generate_random(
        N, r1, r2, ped_r, centerX, centerY, _geoMinX, _geoMaxX, _geoMinY, _geoMaxY, seed
):
    random.seed(seed)
    # biggest radius
    if r1 > r2:
        rmax = r1
        rmin = r2
    else:
        rmax = r2
        rmin = r1

    possible_peds = []
    Rmax = rmax
    while rmax > rmin + ped_r + 0.2:
        # print("-----")
        # print(rmin, rmax)

        rmax -= 2 * ped_r
        delta_theta = 2 * ped_r / rmax
        N_possible = int(np.pi / delta_theta)
        for i in np.arange(0.5, N_possible):
            theta0 = i * delta_theta + np.pi / 2
            # print(f"theta {theta0}, {i}")
            x = centerX + rmax * np.cos(theta0)
            y = centerY + rmax * np.sin(theta0)
            if within_geometry(x, y, _geoMinX, _geoMaxX, _geoMinY, _geoMaxY):
                possible_peds.append((x, y))

    # pl.info(f"Possible positions {len(possible_peds)}")
    if N <= len(possible_peds):
        #print("Before ----")
        #print(possible_peds)
        #print("After ----")
        random.shuffle(possible_peds)
        #print(possible_peds)
        #print("Select ----")
        select_peds = random.sample(possible_peds, N)  # np.pi * random
        #print(select_peds)
    else:
        pl.warning(
            f"Wanted {N} agents between {rmin:.2} and {Rmax:.2}, but only {len(possible_peds)} are possible"
        )
        select_peds = possible_peds

    peds = []
    for x, y in select_peds:
        peds.append((x, y))

    return peds


def main(geometry_file):

    ini_file = ""
    inifiles = []
    if geometry_file:
        # ------ UI
        choice = st.sidebar.radio("Same density for all groups?", ("yes", "no"))
        bash_mode = st.sidebar.radio("Bash mode?", ("no", "yes"), help="no for interactive mode.")
        if bash_mode == "yes":
            n_runs = st.sidebar.number_input("Number of runs", value=10, step=1)
            n_runs = int(n_runs)
        # st.sidebar.write("#### Area 1")
        c1, c2 = st.sidebar.columns((1, 1))
        c1_title = (
            '<p style="font-family:Courier; color:Blue; font-size: 15px;">Circle 1</p>'
        )
        c2_title = (
            '<p style="font-family:Courier; color:Red; font-size: 15px;">Circle 2</p>'
        )
        c2.markdown(c2_title, unsafe_allow_html=True)
        c1.markdown(c1_title, unsafe_allow_html=True)
        rmax = c2.slider("", 1.0, 3.0, 2.0, step=0.1)
        rmin = c1.slider("", 0.1, rmax - 0.5, 0.6, step=0.1)
        # st.sidebar.write("#### Area 2")
        c3_title = (
            '<p style="font-family:Courier; color:Green; font-size: 15px;">Circle 3</p>'
        )
        c4_title = '<p style="font-family:Courier; color:Magenta; font-size: 15px;">Circle 4</p>'
        c1.markdown(c3_title, unsafe_allow_html=True)
        c2.markdown(c4_title, unsafe_allow_html=True)

        rmax2 = c1.slider("", rmax + 0.5, 6.0, 3.0, step=0.1)
        rmin2 = rmax
        # st.sidebar.write("#### Area 3")
        rmax3 = c2.slider("", rmax2 + 0.5, 10.0, 4.0, step=0.1)
        rmin3 = rmax2
        if bash_mode == "no":            
            st.write("#### Motivation state")
            a1, a2, a3 = st.columns((1, 1, 1))
            state1 = a1.number_input("G1", value=0, step=1, min_value=0, max_value=1)
            state2 = a2.number_input("G2", value=0, step=1, min_value=0, max_value=1)
            state3 = a3.number_input("G3", value=0, step=1, min_value=0, max_value=1)

        st.sidebar.write("#### Origin")
        center_x = st.sidebar.number_input("Center x", value=60.0, step=0.1)
        center_y = st.sidebar.number_input("Center y", value=102.0, step=0.1)
        st.sidebar.markdown("### Model parameters: Group 1")
        rped1 = st.sidebar.number_input("r_ped1", value=0.2, step=0.1, format="%.1f")
        c1, c2 = st.sidebar.columns((1, 1))
        v0_1 = c1.number_input("v0_1", value=1.2, step=0.1, format="%.1f")
        sigma_v0_1 = c2.number_input("sigma v0_1", value=0.0, step=0.1, format="%.1f")
        if bash_mode == "no":
            if state1 == 1:
                T_1 = c1.number_input("T_1", value=0.1, step=0.1, format="%.1f")
            if state1 == 0:
                T_1 = c1.number_input("T_1", value=1.3, step=0.1, format="%.1f")

        sigma_T_1 = c2.number_input("sigma T_1", value=0.0, step=0.1, format="%.1f")
        
        st.sidebar.markdown("### Model parameters: Group 2")
        rped2 = st.sidebar.number_input("r_ped2", value=0.2, step=0.1, format="%.1f")

        c1, c2 = st.sidebar.columns((1, 1))
        v0_2 = c1.number_input("v0_2", value=1.2, step=0.1)
        sigma_v0_2 = c2.number_input("sigma v0_2", value=0.0, step=0.1, format="%.1f")
        if bash_mode == "no":
            if state2 == 1:
                T_2 = c1.number_input("T_2", value=0.1, step=0.1, format="%.1f")
            if state2 == 0:
                T_2 = c1.number_input("T_2", value=1.3, step=0.1, format="%.1f")

        sigma_T_2 = c2.number_input("sigma T_2", value=0.0, step=0.1, format="%.1f")
        
        st.sidebar.markdown("### Model parameters: Group 3")
        rped3 = st.sidebar.number_input("r_ped3", value=0.2, step=0.1, format="%.1f")

        c1, c2 = st.sidebar.columns((1, 1))
        v0_3 = c1.number_input("v0_3", value=1.2, step=0.1, format="%.1f")
        sigma_v0_3 = c2.number_input("sigma v0_3", value=0.0, step=0.1, format="%.1f")    
        if bash_mode == "no":
            if state3 == 1:
                T_3 = c1.number_input("T_3", value=0.1, step=0.1, format="%.1f")
            if state3 == 0:
                T_3 = c1.number_input("T_3", value=1.3, step=0.1, format="%.1f")

        sigma_T_3 = c2.number_input("sigma T_3", value=0.0, step=0.1, format="%.1f")

        geo_xml = parseString(geometry_file.getvalue())
        (_geominX, _geomaxX, _geominY, _geomaxY) = geo_limits(geo_xml, unit="m")
        
        # Number of pedestrians
        if choice == "no":
            N1 = st.slider("N1", 10, 50, 5)
            N2 = st.slider("N2", 10, 100, 5)
            N3 = st.slider("N3", 10, 100, 5)
        else:
            N1 = st.slider("N1", 10, 50, 5)
            N2 = N1
            N3 = N1

            A1 = area(
                rmin,
                rmax,
                rped1,
                center_x,
                center_y,
                _geominX,
                _geomaxX,
                _geominY,
                _geomaxY,
            )
            A2 = area(
                rmax,
                rmax2,
                rped2,
                center_x,
                center_y,
                _geominX,
                _geomaxX,
                _geominY,
                _geomaxY,
            )
            A3 = area(
                rmax2,
                rmax3,
                rped3,
                center_x,
                center_y,
                _geominX,
                _geomaxX,
                _geominY,
                _geomaxY,
            )
            N2 = int(A2 / A1 * N1)
            N3 = int(A3 / A1 * N1)
            st.info(
                f"""
            A1: {A1:.2f}, N1: {N1}\n
            A2: {A2:.2f}, N2: {N2}\n
            A3: {A3:.2f}, N3: {N3}"""
            )

        # -----------
        geometry_walls = read_subroom_walls(geo_xml, unit="m")
        if bash_mode == "yes":
            states = list(product([0, 1], [0, 1], [0, 1]))
        else:
            n_runs = 1
            states = [[1, 1, 1]]

        
        for (a, b, c) in states:
            print("states", a,b,c)
            for run in range(n_runs):
                seed = run*100
                peds1 = generate_random(
                    N1,
                    rmin,
                    rmax,
                    rped1,
                    center_x,
                    center_y,
                    _geominX,
                    _geomaxX,
                    _geominY,
                    _geomaxY,
                    seed
                )
                peds2 = generate_random(
                    N2,
                    rmax,
                    rmax2,
                    rped2,
                    center_x,
                    center_y,
                    _geominX,
                    _geomaxX,
                    _geominY,
                    _geomaxY,
                    seed
                )
                peds3 = generate_random(
                    N3,
                    rmax2,
                    rmax3,
                    rped3,
                    center_x,
                    center_y,
                    _geominX,
                    _geomaxX,
                    _geominY,
                    _geomaxY,
                    seed
                )
                if bash_mode == "yes":
                    if a == 1:
                        T_1 = 0.1
                    else:
                        T_1 = 1.3

                    if b == 1:
                        T_2 = 0.1
                    else:
                        T_2 = 1.3
                    
                    if c == 1:
                        T_3 = 0.1
                    else:
                        T_3 = 1.3

                print("---")
                print(run)
                print(peds1)
                print(peds2)
                print(peds3)
                
                # create inifiles
                ini_file = f"ini_run_{run}_state_{a}_{b}_{c}_" + geometry_file.name.split(".")[0] + ".xml"
                #print(ini_file)
                b_xml = write_inifile(
                    rped1,
                    v0_1,
                    sigma_v0_1,
                    T_1,
                    sigma_T_1,
                    peds1,
                    rped2,
                    v0_2,
                    sigma_v0_2,
                    T_2,
                    sigma_T_2,
                    peds2,
                    rped3,
                    v0_3,
                    sigma_v0_3,
                    T_3,
                    sigma_T_3,
                    peds3,
                    ini_file,
                    geometry_file,
                )
                inifiles.append(ini_file)
                
            #print("--", run)
            #print(inifiles, len(inifiles))
    
        # Now we have a bunch of inifiles. Make zip
        # writing files to a zipfile
        if bash_mode == "yes":
            with ZipFile('inifiles.zip','w') as zip:
                # writing each file one by one
                for inifile in inifiles:
                    zip.write(inifile)

            ini_file = ""

        if bash_mode == "no":
            fig = plot_trajectories(geometry_walls, _geominX, _geomaxX, _geominY, _geomaxY)

            # Circle 1
            X, Y = make_circle(center_x, center_y, rmax)
            circle1 = go.Scatter(
                x=X,
                y=Y,
                showlegend=False,
                mode="lines",
                line=dict(color="red", width=2),
            )
            fig.append_trace(circle1, row=1, col=1)
            # Circle 2
            X, Y = make_circle(center_x, center_y, rmin)
            circle2 = go.Scatter(
                x=X,
                y=Y,
                showlegend=False,
                mode="lines",
                line=dict(color="blue", width=2),
            )
            fig.append_trace(circle2, row=1, col=1)
            # Circle 3
            X, Y = make_circle(center_x, center_y, rmax2)
            circle3 = go.Scatter(
                x=X,
                y=Y,
                showlegend=False,
                mode="lines",
                line=dict(color="green", width=2),
            )
            fig.append_trace(circle3, row=1, col=1)
            # Circle 4
            X, Y = make_circle(center_x, center_y, rmax3)
            circle4 = go.Scatter(
                x=X,
                y=Y,
                showlegend=False,
                mode="lines",
                line=dict(color="magenta", width=2),
            )
            fig.append_trace(circle4, row=1, col=1)

            for ped in peds1:
                x, y = ped
                fig.add_shape(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=x - rped1,
                    y0=y - rped1,
                    x1=x + rped1,
                    y1=y + rped1,
                    fillcolor="PaleTurquoise",
                    line_color="LightSeaGreen",
                )

            for ped in peds2:
                x, y = ped
                fig.add_shape(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=x - rped2,
                    y0=y - rped2,
                    x1=x + rped2,
                    y1=y + rped2,
                    fillcolor="Gray",
                    line_color="lightgray",
                )

            for ped in peds3:
                x, y = ped
                fig.add_shape(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=x - rped3,
                    y0=y - rped3,
                    x1=x + rped3,
                    y1=y + rped3,
                    fillcolor="Blue",
                    line_color="LightBlue",
                )

            st.plotly_chart(fig, use_container_width=True)
        
    return ini_file, inifiles


if __name__ == "__main__":

    st.header("**Documentation (click to expand)**")
    with st.expander(""):
        st.write(
            """
    This app creates an inifile that can be used to make JuPedSim-simulations.
     It randomly distributes 3 groups of agents in semi-circular setups.
     Besides, for every group 3 different parameters can be changed:
     - $r\_ped$: The radius of agents
     - $v_0$: The desired speed of agents. The **higher** the more "motivated" are the agents.
     - $T$: The time gap. This parameter can be interpreted as the willingness to close gaps to the neighbors. The **smaller**, the more eager are agents to close gaps.

     When finished tweaking the parameters click on `Download inifile` to download the inifile!
     The simulation with the downloaded file will generate a trajectory file, with the parameter values encoded in its name.
     """
        )
    st.sidebar.image("jupedsim.png", use_column_width=True)
    gh = "https://badgen.net/badge/icon/GitHub?icon=github&label"
    repo = "https://github.com/chraibi/distribute_agents"
    repo_name = f"[![Repo]({gh})]({repo})"
    st.sidebar.markdown(repo_name, unsafe_allow_html=True)
    geometry_file = st.sidebar.file_uploader(
        "🏠 Geometry file ",
        type=["xml"],
        help="Load geometry file",
    )
    ini_file, inifiles = main(geometry_file)
    st.sidebar.write("-----")
    if ini_file:
        with open(ini_file, encoding="utf-8") as f:
            st.download_button("Download inifile",
                               f,
                               file_name=ini_file)
    
    if len(inifiles) >= 2:
        with open("inifiles.zip", "rb") as f:
            st.download_button("Download inifiles",
                               f,
                               file_name="inifiles.zip",
                               mime="application/zip")
    
