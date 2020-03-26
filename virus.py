import bpy
import numpy as np
from mathutils import Vector
import random
import bmesh
from math import pi
import math
from pathlib import Path
import shutil


N_DAYS = 50
FRAMES_PER_DAY = 120
N_FRAMES = N_DAYS * FRAMES_PER_DAY


def arrow(length, width, head_length, head_width, thickness, collection, zero_pad=False):
    verts = [[0, 0.5 * width, 0],
             [length - head_length, 0.5 * width, 0],
             [length - head_length, 0.5 * head_width, 0],
             [length, 0, 0],
             [length - head_length, -0.5 * head_width, 0],
             [length - head_length, -0.5 * width, 0],
             [0, -0.5 * width, 0]]

    if zero_pad:
        verts[0][0] -= 0.5 * width
        verts[-1][0] -= 0.5 * width

    faces = [(0, 1, 5, 6),
             (1, 2, 3, 4, 5)]

    mesh = bpy.data.meshes.new("arrow")
    obj = bpy.data.objects.new(mesh.name, mesh)
    collection.objects.link(obj)
    mesh.from_pydata(verts, [], faces)

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.solidify(bm, geom=bm.faces, thickness=thickness)
    bm.to_mesh(mesh)
    bm.free()
    for vertex in obj.data.vertices:
        vertex.co[2] -= 0.5 * thickness
    obj.data.update()

    return obj


def make_curve(coordinates, collection, bevel_depth=0.01, auto_reduce=False):
    # create the Curve Datablock
    curve = bpy.data.curves.new('line', type='CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 2
    curve.bevel_depth = bevel_depth

    # map coords to spline
    spline = curve.splines.new('NURBS')
    spline.points.add(len(coordinates) - 1)
    print(len(spline.points))
    print(len(coordinates))
    if auto_reduce:
        reduced_coordinates = [coordinates[0]]
        for i in range(1, len(coordinates)):
            if coordinates[i, 1] != coordinates[i-1, 1]:
                reduced_coordinates.append(coordinates[i])
        coordinates = reduced_coordinates

    for i, coord in enumerate(coordinates):
        x, y, z = coord
        spline.points[i].co = (x, y, z, 1)
    spline.order_u = 2

    # create Object
    curveOB = bpy.data.objects.new('myCurve', curve)
    collection.objects.link(curveOB)
    return curveOB


def make_cuboid(name, collection, shape=(1, 1, 1)):
    mesh = bpy.data.meshes.new(name)
    cube = bpy.data.objects.new(name, mesh)
    collection.objects.link(cube)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(mesh)
    bm.free()
    for vertex in cube.data.vertices:
        for i in range(3):
            vertex.co[i] *= shape[i]

    return cube


def make_text(content, collection, name="text"):
    curve = bpy.data.curves.new(type="FONT", name="text_curve")
    text = bpy.data.objects.new(name, curve)
    text.data.body = content
    collection.objects.link(text)
    return text


class Axis:
    def __init__(self, length, extent, arrow_scale, tick_locations, collection, parent=None):
        self.extent = extent
        self.collection = collection
        self.arrow = arrow(length, arrow_scale, arrow_scale * 2, arrow_scale * 2, arrow_scale, self.collection,
                           zero_pad=True)

        self.ticks = []
        self.tick_labels = []
        for location in tick_locations:
            transformed_tick_location = location / (self.extent[1] - self.extent[0]) * length
            tick = make_cuboid("tick", self.collection, (0.5 * arrow_scale, 2 * arrow_scale, arrow_scale))
            tick.parent = self.arrow
            tick.location[0] += transformed_tick_location
            tick.location[1] -= 0.5 * arrow_scale

            label = make_text(str(location), self.collection, name="tick_label")
            label.select_set(True)
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
            label.select_set(False)
            label.location = (0, 0, 0)
            label.parent = tick
            label.location[1] -= 5 * arrow_scale
            label.scale *= arrow_scale * 5

            self.ticks.append(tick)
            self.tick_labels.append(label)


class Figure:
    def __init__(self, data, tick_locations, shape=(1, 1, -1), colors=None):
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)

        if data.shape[-1] == 2:
            data = np.concatenate((data, np.zeros(data.shape[:2] + (1,))), axis=2)
        self.shape = shape
        self.extent = np.array([[np.min(data[:, :, i]), np.max(data[:, :, i])] for i in range(data.shape[2])])
        self.collection = add_collection("figure", clear=True)
        self.origin = bpy.data.objects.new("origin", None)
        self.collection.objects.link(self.origin)
        self.origin.empty_display_type = 'ARROWS'
        # bpy.ops.mesh.primitive_plane_add()
        # self.background = bpy.context.active_object

        self.axes = []
        for i in range(len(self.shape)):
            if self.shape[i] != -1:
                axis = Axis(length=self.shape[i],
                            extent=self.extent[i],
                            arrow_scale=self.shape[1] * 0.02,
                            tick_locations=tick_locations[i],
                            collection=self.collection)
                axis.arrow.parent = self.origin
                self.axes.append(axis)

            else:
                self.axes.append(None)
        if self.axes[1] is not None:
            self.axes[1].arrow.rotation_euler = (pi, 0, pi / 2)
            for label in self.axes[1].tick_labels:
                label.rotation_euler = (0, pi, -pi / 2)

        if self.axes[2] is not None:
            self.axes[2].arrow.rotation_euler = (0, pi / 2, 0)

        ranges = np.expand_dims(np.expand_dims(self.extent[:, 1] - self.extent[:, 0], axis=0), axis=0)
        ranges[ranges == 0] = 1
        transformed_data = data / ranges * np.expand_dims(np.expand_dims(np.array(self.shape), axis=0), axis=0)

        self.lines = [make_curve(transformed_data[i], self.collection, bevel_depth=self.shape[1] * .01)
                      for i in range(transformed_data.shape[0])]
        for i, line in enumerate(self.lines):
            line.parent = self.origin
            if colors is not None:
                material = bpy.data.materials.new(name="line")
                material.diffuse_color = colors[i]
                line.data.materials.append(material)


def copy_ob(ob, parent, collection):
    # copy ob
    copy = ob.copy()
    copy.parent = parent
    copy.matrix_parent_inverse = ob.matrix_parent_inverse.copy()
    # copy particle settings
    for ps in copy.particle_systems:
        ps.settings = ps.settings.copy()
    collection.objects.link(copy)
    return copy


def tree_copy(ob, collection, levels=3):
    def recurse(ob, parent, depth):
        if depth > levels:
            return
        copy = copy_ob(ob, parent, collection)

        for child in ob.children:
            recurse(child, copy, depth + 1)
        return copy

    return recurse(ob, ob.parent, 0)


def circle_place(n_places, taken_places=[], free_radius=1, radius=1.0, n_dims=2):
    # print("********************************************************************")
    positions = []
    for i in range(n_places):
        while True:
            while True:
                test_coordinates = np.random.uniform(-radius, radius, size=n_dims)
                if np.linalg.norm(test_coordinates) < radius:
                    break
            # print(test_coordinates)
            conflict = False
            for taken_place in taken_places + positions:
                distance = np.linalg.norm(taken_place[0] - test_coordinates)
                # print(distance)
                if distance < taken_place[1]:
                    conflict = True
                    break
            if not conflict:
                positions.append((test_coordinates, free_radius))
                break
    return positions


def add_collection(name, activate=True, clear=False):
    if name not in [c.name for c in bpy.data.collections]:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    else:
        collection = bpy.data.collections[name]

    if activate:
        layer_collection = bpy.context.view_layer.layer_collection.children[collection.name]
        bpy.context.view_layer.active_layer_collection = layer_collection

    if clear:
        while collection.objects:
            bpy.data.objects.remove(collection.objects[0])
    return collection


class Factory:
    def __init__(self, collection, blend_object=None):
        base_factory = bpy.data.collections["object_stash"].all_objects["factory"]
        if blend_object is None:
            self.b = tree_copy(base_factory, collection)
        else:
            self.b = blend_object
        self.n_workers = 0
        self.workers = []
        self.capacity = 30
        self.rooms = [Room(self) for i in range(3)]


class Store:
    def __init__(self, collection, blend_object=None):
        base_store = bpy.data.collections["object_stash"].all_objects["store"]
        if blend_object is None:
            self.b = tree_copy(base_store, collection)
        else:
            self.b = blend_object
        self.capacity = 10
        self.rooms = [Room(self)]

    def enter(self, marble):
        if self.capacity > 0:
            marble.current_room.leave(marble)
            self.rooms[0].enter(marble)
            marble.current_room = self.rooms[0]
            marble.b.location = self.b.location
            marble.b.location[0] += 10 - self.capacity
            marble.b.location[1] -= 0.2
            self.capacity -= 1
            return True
        else:
            return False

    def reset(self):
        self.capacity = 10


class House:
    def __init__(self, collection, blend_object=None):
        base_house = bpy.data.collections["object_stash"].all_objects["house"]
        if blend_object is None:
            self.b = tree_copy(base_house, collection)
        else:
            self.b = blend_object
        self.rooms = [Room(self) for i in range(6)]


class Statistics:
    def __init__(self):
        self.n_infected = 0
        self.n_immune = 0
        self.n_marbles = 0
        self.n_infectable = 0

        self.infections = np.zeros(N_FRAMES)
        self.heals = np.zeros(N_FRAMES)

    def get_infected_timeline(self):
        return np.cumsum(self.infections) - np.cumsum(self.heals)

    def get_immune_timeline(self):
        return np.cumsum(self.heals)

    def get_infectable_timeline(self):
        return self.n_marbles - np.cumsum(self.infections)


class Marble:
    statistics = Statistics()

    def __init__(self, collection, infection_rate=0.03):
        self.infection_rate = infection_rate
        base_marble = bpy.data.collections["object_stash"].all_objects["marble"]
        self.b = base_marble.copy()
        # self.b = bpy.data.objects.new(base_marble.name, base_marble.data.copy())
        self.b.data = base_marble.data.copy()
        self.colors = [(0.7, 0.01, 0.01, 1),
                       (0.014, 0.012, 1, 1),
                       (1, 0.7, 0.015, 1)]

        material = bpy.data.materials.new(name="marble")
        material.diffuse_color = self.colors[2]
        self.b.active_material = material
        # self.b.data.materials.append(material)
        self.b.animation_data_create()
        collection.objects.link(self.b)
        self.home = None
        self.home_room = None
        self.work_place = None
        self.work_room = None
        self.current_room = self.home_room
        self.sick = False
        self.immune = False
        self.quarantine = False
        self.home_office = False
        self.outbreak = None
        self.caused_infections = 0
        self.statistics.n_marbles += 1
        self.statistics.n_infectable += 1

    def go_work(self):
        if self.current_room is not self.work_room:
            self.b.location = self.work_place
            self.current_room.leave(self)
            self.work_room.enter(self)
            self.current_room = self.work_room

    def go_home(self):
        if self.current_room is not self.home_room:
            self.b.location = self.home
            self.current_room.leave(self)
            self.home_room.enter(self)
            self.current_room = self.home_room

    def infect(self, day, frame):
        if not (self.sick or self.immune):
            self.sick = True
            self.outbreak = day
            self.statistics.n_infected += 1
            self.statistics.n_infectable -= 1
            self.statistics.infections[frame] += 1

            self.b.active_material.keyframe_insert(data_path="diffuse_color", frame=frame)
            self.b.active_material.diffuse_color = self.colors[0]
            self.b.active_material.keyframe_insert(data_path="diffuse_color", frame=frame + 5)
            # mix_node = self.b.active_material.node_tree.nodes.get("Mix Shader")
            # mix_node.inputs[0].keyframe_insert("default_value", frame=frame)
            # mix_node.inputs[0].default_value = 1
            # mix_node.inputs[0].keyframe_insert("default_value", frame=frame+5)
            return True
        else:
            return False

    def heal(self, frame):
        self.sick = False
        self.statistics.n_infected -= 1
        self.statistics.heals[frame] += 1
        # if random.random() < 0.5:
        if True:
            self.immune = True
            self.statistics.n_immune += 1
            self.b.active_material.keyframe_insert(data_path="diffuse_color", frame=frame)
            self.b.active_material.diffuse_color = self.colors[1]
            self.b.active_material.keyframe_insert(data_path="diffuse_color", frame=frame + 5)
            # mix_node = self.b.active_material.node_tree.nodes.get("Mix Shader.001")
            # mix_node.inputs[0].keyframe_insert("default_value", frame=frame)
            # mix_node.inputs[0].default_value = 1
            # mix_node.inputs[0].keyframe_insert("default_value", frame=frame + 5)
        # else:
        #     self.statistics.n_infectable += 1
        #     mix_node = self.b.active_material.node_tree.nodes.get("Mix Shader.001")
        #     mix_node.inputs[0].keyframe_insert("default_value", frame=frame)
        #     mix_node.inputs[0].default_value = 0
        #     mix_node.inputs[0].keyframe_insert("default_value", frame=frame + 5)
        #     mix_node = self.b.active_material.node_tree.nodes.get("Mix Shader")
        #     mix_node.inputs[0].keyframe_insert("default_value", frame=frame)
        #     mix_node.inputs[0].default_value = 0
        #     mix_node.inputs[0].keyframe_insert("default_value", frame=frame + 5)

    def spread(self, day, frame):
        if self.sick:
            for marble in self.current_room.present:
                if random.random() < self.infection_rate:
                    if marble.infect(day, frame):
                        self.caused_infections += 1

    def get_tested(self, frame):
        if self.sick and not self.quarantine:
            self.b.keyframe_insert(data_path="location", frame=frame - 5)
            self.b.active_material.keyframe_insert(data_path="diffuse_color", frame=frame - 5)
            self.b.active_material.diffuse_color = (0, 1, 0, 1)
            self.b.location[1] -= 0.5
            self.b.keyframe_insert(data_path="location", frame=frame)
            self.b.active_material.keyframe_insert(data_path="diffuse_color", frame=frame)
            self.b.active_material.diffuse_color = self.colors[0]
            self.b.location[1] += 0.5
            self.b.keyframe_insert(data_path="location", frame=frame + 5)
            self.b.active_material.keyframe_insert(data_path="diffuse_color", frame=frame + 5)
            self.quarantine = True
            for marble in self.home_room.associates:
                marble.quarantine = True

            for marble in self.work_room.associates:
                marble.get_tested(frame)


class Room:
    def __init__(self, building):
        self.building = building
        self.associates = []
        self.present = []

    def enter(self, marble):
        self.present.append(marble)

    def leave(self, marble):
        self.present.remove(marble)


def remove_orphans():
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

    for block in bpy.data.actions:
        if block.users == 0:
            bpy.data.actions.remove(block)

    for block in bpy.data.curves:
        if block.users == 0:
            bpy.data.curves.remove(block)


def setup_city(size, radius):
    n_factories = size
    n_stores = n_factories
    n_houses = 2 * n_factories
    # rotation_sigma = pi/64

    factory_collection = add_collection("factories", clear=True)
    factory_positions = circle_place(n_factories, radius=radius, free_radius=10)
    factories = []
    for factory_position in factory_positions:
        factory = Factory(factory_collection)
        factory.b.location[:2] = factory_position[0]
        # factory.b.rotation_euler[2] += random.gauss(0, rotation_sigma)
        factories.append(factory)
    print("placed factories")

    store_collection = add_collection("stores", clear=True)
    store_positions = circle_place(n_stores, taken_places=factory_positions, radius=radius, free_radius=10)
    stores = []
    for store_position in store_positions:
        store = Store(store_collection)
        stores.append(store)
        store.b.location[:2] = store_position[0]
        # store.b.rotation_euler[2] += random.gauss(0, rotation_sigma)
    print("places stores")

    houses_collection = add_collection("houses", clear=True)
    house_positions = circle_place(n_houses, taken_places=factory_positions + store_positions,
                                   radius=radius, free_radius=5)
    houses = []
    for house_position in house_positions:
        house = House(houses_collection)
        houses.append(house)
        house.b.location[:2] = house_position[0]
        # house.b.rotation_euler[2] += random.gauss(0, rotation_sigma)
    print("placed houses")

    return factories, stores, houses


def load_city():
    factory_collection = bpy.data.collections["factories"]
    factories = []
    for factory in factory_collection.objects:
        if "factory" in factory.name:
            factories.append(Factory(factory_collection, factory))

    store_collection = bpy.data.collections["stores"]
    stores = []
    for store in store_collection.objects:
        if "store" in store.name:
            stores.append(Store(store_collection, store))

    house_collection = bpy.data.collections["houses"]
    houses = []
    for house in house_collection.objects:
        if "house" in house.name:
            houses.append(House(house_collection, house))

    return factories, stores, houses


def setup_marbles(factories, houses, infection_rate=0.03):
    marbles_collection = add_collection("marbles", clear=True)
    marbles = []
    for house in houses:
        for z, room in enumerate(house.rooms):
            n_occupants = np.random.choice([1, 2, 3, 4], p=[.4, .35, .15, .10])
            for i in range(n_occupants):
                marble = Marble(marbles_collection, infection_rate)
                marble.b.location = house.b.location + Vector([i, -0.2, z])
                marble.home = marble.b.location[:]
                marble.home_room = room
                marble.current_room = marble.home_room
                room.associates.append(marble)
                room.present.append(marble)
                marbles.append(marble)

    random.shuffle(marbles)
    print("created marbles")

    assert sum([f.capacity for f in factories]) >= len(marbles)
    for marble in marbles:
        while True:
            factory = random.choice(factories)
            # print(factory.n_workers, factory.capacity)
            if factory.n_workers < factory.capacity:
                factory.workers.append(marble)
                factory.n_workers += 1
                break
    for factory in factories:
        x = 0
        z = 0
        for worker in factory.workers:
            worker.work_place = factory.b.location + Vector((x, -0.2, z))
            worker.work_room = factory.rooms[z]
            factory.rooms[z].associates.append(worker)
            x += 1
            if x == 10:
                x = 0
                z += 1
    print("distributed marbles")

    return marbles


def simulate(marbles, stores, quarantine_rule, home_office_quota, start_infections):
    frame_i = 0
    for i in range(start_infections):
        marbles[i].infect(0, frame_i)

    for day in range(N_DAYS):
        print(marbles[0].statistics.n_infected)
        # print("\r", day, end="")
        if home_office_quota:
            if day == 7:
                for marble in marbles:
                    if random.random() < home_office_quota:
                        marble.home_office = True

        for marble in marbles:
            if quarantine_rule and marble.quarantine:
                lift = True
                for flatmate in marble.home_room.associates:
                    if flatmate.sick:
                        lift = False
                if lift:
                    for flatmate in marble.home_room.associates:
                        flatmate.quarantine = False

            if marble.sick:
                if day - marble.outbreak > 14:
                    marble.heal(frame_i)
            marble.spread(day, frame_i)

        for marble in marbles:
            marble.b.keyframe_insert(data_path="location", frame=frame_i + 10)
            if ((not marble.sick) or (day - marble.outbreak < 6)) and \
                    (not (marble.quarantine and quarantine_rule)) and (not marble.home_office):
                marble.go_work()
            marble.b.keyframe_insert(data_path="location", frame=frame_i + 30)
        for marble in marbles:
            marble.spread(day, frame_i + 35)
        if quarantine_rule:
            for marble in marbles:
                if marble.sick and day - marble.outbreak >= 6:
                    marble.get_tested(frame_i+41)

        shoppers = np.array([False] * len(marbles))
        shoppers[random.sample(range(len(marbles)), int(len(marbles) / 3))] = True
        for i, marble in enumerate(marbles):
            marble.b.keyframe_insert(data_path="location", frame=frame_i + 50)
            if ((not marble.sick) or (day - marble.outbreak < 6)) and \
                    (not (marble.quarantine and quarantine_rule)):
                if shoppers[i]:
                    while True:
                        store = random.choice(stores)
                        if store.enter(marble):
                            break
                else:
                    marble.go_home()
            else:
                marble.go_home()

            marble.b.keyframe_insert(data_path="location", frame=frame_i + 70)
        for marble in marbles:
            marble.spread(day, frame_i + 80)
        for store in stores:
            store.reset()

        for marble in marbles:
            marble.b.keyframe_insert(data_path="location", frame=frame_i + 90)
            marble.go_home()
            marble.b.keyframe_insert(data_path="location", frame=frame_i + 110)
        for marble in marbles:
            marble.spread(day, frame_i + 110)

        frame_i += FRAMES_PER_DAY
    print("")


def analysis(marbles, save_path):
    statistics = marbles[0].statistics
    figure_step = int(FRAMES_PER_DAY / 4)
    fig_frames = np.arange(N_FRAMES)[::figure_step]
    times_in_weeks = fig_frames / FRAMES_PER_DAY / 7
    infected_data = np.stack((times_in_weeks,
                              statistics.get_infected_timeline()[
                              ::figure_step] / statistics.n_marbles * 100),
                             axis=1)
    immune_data = np.stack((times_in_weeks,
                            statistics.get_immune_timeline()[::figure_step] / statistics.n_marbles * 100),
                           axis=1)
    infectable_data = np.stack((times_in_weeks,
                                statistics.get_infectable_timeline()[
                                ::figure_step] / statistics.n_marbles * 100),
                               axis=1)

    tick_locations = [[0, 1, 2, 3, 4, 5, 6],
                      [0, 25, 50, 75],
                      []]

    colors = [(0.7, 0.01, 0.01, 1),
              (0.014, 0.012, 1, 1),
              (1, 0.7, 0.015, 1)]

    fig = Figure(np.stack((infected_data, immune_data, infectable_data)), tick_locations, colors=colors)
    fig.origin.rotation_euler[0] = pi / 2
    fig.origin.location[2] = 34.571
    fig.origin.location[0] = 14.374
    fig.origin.scale *= 25.93

    # fig.lines[0].active_material

    for frame in fig_frames:
        for line in fig.lines:
            line.data.bevel_factor_end = frame / N_FRAMES
            line.data.keyframe_insert(data_path='bevel_factor_end', frame=frame)

    # for line in fig.lines:
    #     fcurves = line.data.animation_data.action.fcurves
    #     for fcurve in fcurves:
    #         for kf in fcurve.keyframe_points:
    #             kf.interpolation = 'CONSTANT'

    outbreak = []
    caused_infections = []
    for marble in marbles:
        if marble.outbreak is not None:
            caused_infections.append(marble.caused_infections)
            outbreak.append(marble.outbreak)
    caused_infections_data = np.array([outbreak, caused_infections]).T
    np.save(save_path / 'caused_infections_data.npy', caused_infections_data)

    np.save(save_path / 'infected_data.npy', infected_data)
    np.save(save_path / 'immune_data.npy', immune_data)
    np.save(save_path / 'infectable_data.npy', infectable_data)

    np.save(save_path / 'infected_frame_res.npy', statistics.get_infected_timeline())
    np.save(save_path / 'frames_per_day.npy', FRAMES_PER_DAY)


def run(size, radius, quarantine_rule, home_office_quota, start_infections,
        create_city=True, infection_rate=0.03):
    print("*****************************************************")
    remove_orphans()
    print("removed orphans")
    if create_city:
        factories, stores, houses = setup_city(size, radius)
    else:
        factories, stores, houses = load_city()

    print(houses)
    marbles = setup_marbles(factories, houses, infection_rate=infection_rate)
    print(len(marbles), " marbles created")

    simulate(marbles, stores, quarantine_rule, home_office_quota, start_infections)
    analysis(marbles, Path(bpy.data.filepath).parent)

    print("Population: ", marbles[0].statistics.n_marbles)
    print("Immune: ", marbles[0].statistics.n_immune)
    print("Infected: ", marbles[0].statistics.n_infected)


def multi_run(number, size, radius, quarantine_rule, home_office_quota, start_infections,
              infection_rate=0.03):

    for i in range(number):
        print("*****************************************************")
        remove_orphans()
        print("removed orphans")
        factories, stores, houses = setup_city(size, radius)

        marbles = setup_marbles(factories, houses, infection_rate)

        simulate(marbles, stores, quarantine_rule, home_office_quota, start_infections=start_infections)

        save_path = Path(f"D:/render/virus/runs/size{size}/q{str(quarantine_rule)}h{home_office_quota}/run{i}")
        save_path.mkdir(parents=True, exist_ok=True)
        analysis(marbles, save_path)

        print("Population: ", marbles[0].statistics.n_marbles)
        print("Immune: ", marbles[0].statistics.n_immune)
        print("Infected: ", marbles[0].statistics.n_infected)

        current_path = bpy.data.filepath
        bpy.ops.wm.save_as_mainfile(filepath=str(save_path / "virus_run.blend"))
        bpy.ops.wm.open_mainfile(filepath=current_path)

        shutil.copy(Path("X:/Google Drive/blender/virus/viruslib/register_handlers.py"), save_path)
        marbles[0].statistics.__init__()  # to reset statistics


