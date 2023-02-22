import numpy as np
import random
from .sketch import Profile
from .macro import *
from .math_utils import *


class CoordSystem(object):
    """Local coordinate system for sketch plane."""

    def __init__(self, origin, theta, phi, gamma, y_axis=None,is_numerical=False):
        self.origin = origin
        self._theta = theta  # 0~pi
        self._phi = phi     # -pi~pi
        self._gamma = gamma  # -pi~pi
        self._y_axis = y_axis  # (theta, phi)
        self.is_numerical = is_numerical

    @property
    def normal(self):
        return polar2cartesian([self._theta, self._phi])

    @property
    def x_axis(self):
        normal_3d, x_axis_3d = polar_parameterization_inverse(
            self._theta, self._phi, self._gamma)
        return x_axis_3d

    @property
    def y_axis(self):
        if self._y_axis is None:
            return np.cross(self.normal, self.x_axis)
        return polar2cartesian(self._y_axis)

    @staticmethod
    def from_dict(refPlane, transform):
        # origin = np.array(
        #     [refPlane["origin"]["x"], refPlane["origin"]["y"], refPlane["origin"]["z"]])
        normal_3d = np.array(
            [refPlane["normal"]["x"], refPlane["normal"]["y"], refPlane["normal"]["z"]])
        x_axis_3d = unit_vector(np.array(transform[0:3]))
        y_axis_3d = unit_vector(np.array(transform[3:6]))
        z_axis_3d = unit_vector(np.array(transform[6:9]))
        origin=np.array(transform[9:12])
        # Check if normal and z_axis are same
        #check_distance(normal_3d, z_axis_3d)
        theta, phi, gamma = polar_parameterization(normal_3d, x_axis_3d)
        y_axis=cartesian2polar(y_axis_3d)
        return CoordSystem(origin, theta, phi, gamma, y_axis=y_axis)

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        origin = vec[:3]
        theta, phi, gamma = vec[3:]
        system = CoordSystem(origin, theta, phi, gamma)
        if is_numerical:
            system.denumericalize(n)
        return system

    def __str__(self):
        return "origin: {}, normal: {}, x_axis: {}, y_axis: {}".format(
            self.origin.round(4), self.normal.round(4), self.x_axis.round(4), self.y_axis.round(4))

    def transform(self, translation, scale):
        self.origin = (self.origin + translation) * scale

    def numericalize(self, n=256):
        """NOTE: shall only be called after normalization"""
        # assert np.max(self.origin) <= 1.0 and np.min(self.origin) >= -1.0 # TODO: origin can be out-of-bound!
        self.origin = ((self.origin + 1.0) / 2 *
                       n).round().clip(min=0, max=n-1).astype(np.int32)
        tmp = np.array([self._theta, self._phi, self._gamma])
        self._theta, self._phi, self._gamma = ((tmp / np.pi + 1.0) / 2 * n).round().clip(
            min=0, max=n-1).astype(np.int32)
        self.is_numerical = True

    def denumericalize(self, n=256):
        self.origin = self.origin / n * 2 - 1.0
        tmp = np.array([self._theta, self._phi, self._gamma])
        self._theta, self._phi, self._gamma = (tmp / n * 2 - 1.0) * np.pi
        self.is_numerical = False

    def to_vector(self):
        return np.array([*self.origin, self._theta, self._phi, self._gamma])


class Extrude(object):
    """Single extrude operation with corresponding a sketch profile.
    NOTE: only support single sketch profile. Extrusion with multiple profiles is decomposed."""

    def __init__(self, profile: Profile, sketch_plane: CoordSystem,
                 operation, 
                 extent_type, extent_one, extent_two, 
                 sketch_pos=None, sketch_size=None):
        """
        Args:
            profile (Profile): normalized sketch profile
            sketch_plane (CoordSystem): coordinate system for sketch plane
            operation (int): index of EXTRUDE_OPERATIONS, see macro.py
            extent_type (int): index of EXTENT_TYPE, see macro.py
            extent_one (float): extrude distance in normal direction (NOTE: it's negative in some data)
            extent_two (float): extrude distance in opposite direction
            extent_start (float): Start coordinate for extrusion
            extent_end_one (float): End coordinate for extrusion one
            extent_end_two(float): End coordinate for extrusion two
            sketch_pos (np.array): the global 3D position of sketch starting point
            sketch_size (float): size of the sketch
        """
        self.profile = profile  # normalized sketch
        self.sketch_plane = sketch_plane
        self.operation = operation
        self.extent_type = extent_type
        self.extent_one = extent_one
        self.extent_two = extent_two
        # self.extent_start = extent_start
        # self.extent_end_one = extent_end_one
        # self.extent_end_two = extent_end_two
        self.sketch_pos = sketch_pos
        self.sketch_size = sketch_size


    @staticmethod
    def from_dict(all_stat, extrude_index, sketch_dim=256):
        """construct Extrude from json data

        Args:
            all_stat (dict): all json data
            extrude_index (str): entity index for this extrude
            sketch_dim (int, optional): sketch normalization size. Defaults to 256.

        Returns:
            list: one or more Extrude instances
        """

        extrude_entity = all_stat["entities"][extrude_index]['feature']
        #assert extrude_entity["start_extent"]["type"] == "ProfilePlaneStartDefinition"

        all_skets = []
        # Number of sketches(in most cases it's 1)
        n = len(extrude_entity['extrude']["references"])
        for i in range(n):
            sketch_id = extrude_entity["extrude"]["references"][i]
            try:
                sket_entity = all_stat["entities"][Extrude.get_sketch_index(
                    all_stat, sketch_id)]['sketch']
            except:
                #print(f"Key Mismatch for Extrusion and Sketch: Path {all_stat['path']}, Sketch ID {sketch_id}")
                continue
            sket_profile = Profile.from_dict(sket_entity)
            if sket_profile is None:
                continue
            try:
                sket_plane = CoordSystem.from_dict(
                    sket_entity["refPlane"], sket_entity["transform"]["data"])
            except Exception as e:
                #print(f"Problem:{e}\n")
                #print("Path:",all_stat['path'])
                continue
            # normalize profile
            point = sket_profile.start_point
            # Linear Transformation of points
            sket_pos = point[0] * sket_plane.x_axis + point[1] * sket_plane.y_axis+ sket_plane.origin
            sket_size = sket_profile.bbox_size
            #sket_profile.normalize(sketch_dim)
            all_skets.append((sket_profile, sket_plane, sket_pos, sket_size))

        operation = EXTRUDE_OPERATIONS.index(extrude_entity["boolean"])

        # Get Extent Direction and amount
        extent_one = Extrude.get_extent_amount(
            extrude_entity['extrude']['refAxis'][0])
        extent_start = np.array(
            list(extrude_entity['extrude']['refAxis'][0]['start'].values()))
        extent_end_one = np.array(
            list(extrude_entity['extrude']['refAxis'][0]['end'].values()))
        if len(extrude_entity['extrude']['refAxis']) == 1:
            extent_type = EXTENT_TYPE.index("OneSideFeatureExtentType")
            extent_end_two = None
            extent_two = 0.0
        elif len(extrude_entity['extrude']['refAxis']) == 2:
            extent_type =EXTENT_TYPE.index("TwoSidesFeatureExtentType")
            extent_two = Extrude.get_extent_amount(
                extrude_entity['extrude']['refAxis'][1])
            extent_end_two = np.array(
                list(extrude_entity['extrude']['refAxis'][1]['end'].values()))
        else:
            raise Exception(
                "Extrusion in more than 2 axes is not yet supported")
        

        if operation == EXTRUDE_OPERATIONS.index("NewBodyFeatureOperation"):
            all_operations = [
                operation] + [EXTRUDE_OPERATIONS.index("JoinFeatureOperation")] * (n - 1)
        else:
            all_operations = [operation] * n

        if len(all_skets)>0:
            return [Extrude(all_skets[i][0], all_skets[i][1], all_operations[i], 
                            extent_type, extent_one, 
                            extent_two,
                            all_skets[i][2], all_skets[i][3]) for i in range(n)]
        else:
            return None
    @staticmethod
    def get_extent_amount(refAxis):
        start = np.array(
            [refAxis['start']["x"], refAxis['start']["y"], refAxis['start']["z"]])
        end = np.array([refAxis['end']["x"], refAxis['end']
                       ["y"], refAxis['end']["z"]])
        if end[0]=="NaN":
            end=np.array([1,1,1])
        return l1_distance(start, end)

    @staticmethod
    def get_sketch_index(all_stat, sketch_id):
        for index,ent in enumerate(all_stat["entities"]):
            key_=list(ent.keys())[0]
            if ent[key_]['uuid'] == sketch_id:
                return index
        return None

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        """vector representation: commands [SOL, ..., SOL, ..., EXT]"""
        assert vec[-1][0] == EXT_IDX and vec[0][0] == SOL_IDX
        profile_vec = np.concatenate([vec[:-1], EOS_VEC[np.newaxis]])
        profile = Profile.from_vector(profile_vec, is_numerical=is_numerical)
        ext_vec = vec[-1][-N_ARGS_EXT:]

        sket_pos = ext_vec[N_ARGS_PLANE:N_ARGS_PLANE + 3]
        sket_size = ext_vec[N_ARGS_PLANE + N_ARGS_TRANS - 1]
        sket_plane = CoordSystem.from_vector(
            np.concatenate([sket_pos, ext_vec[:N_ARGS_PLANE]]))
        ext_param = ext_vec[-N_ARGS_EXT_PARAM:]

        res = Extrude(profile, sket_plane, int(ext_param[2]), int(ext_param[3]), ext_param[0], ext_param[1],
                      sket_pos, sket_size)
        if is_numerical:
            res.denumericalize(n)
        return res

    def __str__(self):
        s = "Sketch-Extrude pair:"
        s += "\n  -" + str(self.sketch_plane)
        s += "\n  -sketch position: {}, sketch size: {}".format(
            self.sketch_pos.round(4), self.sketch_size.round(4))
        s += "\n  -operation:{}, type:{}, extent_one:{}, extent_two:{}".format(
            self.operation, self.extent_type, self.extent_one.round(4), self.extent_two.round(4))
        s += "\n  -" + str(self.profile)
        return s
    
    def normalize(self):
        self.profile.normalize(256)
        pass

    def transform(self, translation, scale):
        """linear transformation"""
        # self.profile.transform(np.array([0, 0]), scale)
        self.sketch_plane.transform(translation, scale)
        self.extent_one *= scale
        self.extent_two *= scale
        self.sketch_pos = (self.sketch_pos + translation) * scale
        self.sketch_size *= scale

    def numericalize(self, n=256):
        """quantize the representation.
        NOTE: shall only be called after CADSequence.normalize (the shape lies in unit cube, -1~1)"""
        #assert -2.0 <= self.extent_one <= 2.0 and -2.0 <= self.extent_two <= 2.0
        self.profile.numericalize(n)
        self.sketch_plane.numericalize(n)
        self.extent_one = ((self.extent_one + 1.0) / 2 *
                           n).round().clip(min=0, max=n-1).astype(np.int32)
        self.extent_two = ((self.extent_two + 1.0) / 2 *
                           n).round().clip(min=0, max=n-1).astype(np.int32)
        self.operation = int(self.operation)
        self.extent_type = int(self.extent_type)
        self.sketch_pos = ((self.sketch_pos + 1.0) / 2 *
                           n).round().clip(min=0, max=n-1).astype(np.int32)
        self.sketch_size = (self.sketch_size / 2 *
                            n).round().clip(min=0, max=n-1).astype(np.int32)

    def denumericalize(self, n=256):
        """de-quantize the representation."""
        self.extent_one = self.extent_one / n * 2 - 1.0
        self.extent_two = self.extent_two / n * 2 - 1.0
        self.sketch_plane.denumericalize(n)
        self.sketch_pos = self.sketch_pos / n * 2 - 1.0
        self.sketch_size = self.sketch_size / n * 2

        self.operation = self.operation
        self.extent_type = self.extent_type

    def flip_sketch(self, axis):
        self.profile.flip(axis)
        self.profile.normalize()

    def to_vector(self, max_n_loops=6, max_len_loop=15, pad=True):
        """vector representation: commands [SOL, ..., SOL, ..., EXT]"""
        profile_vec = self.profile.to_vector(
            max_n_loops, max_len_loop, pad=False)
        if profile_vec is None:
            return None
        sket_plane_orientation = self.sketch_plane.to_vector()[3:]
        ext_param = list(sket_plane_orientation) + list(self.sketch_pos) + [self.sketch_size] + \
            [self.extent_one, self.extent_two, self.operation, self.extent_type]
        ext_vec = np.array([EXT_IDX, *[PAD_VAL] * N_ARGS_SKETCH, *ext_param])
        # NOTE: last one is EOS
        vec = np.concatenate(
            [profile_vec[:-1], ext_vec[np.newaxis], profile_vec[-1:]], axis=0)
        if pad:
            pad_len = max_n_loops * max_len_loop - vec.shape[0]
            vec = np.concatenate(
                [vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return vec

    def sample_points(self):
        """
        Sample points from the profile/sketch and transform it to the CAD plane.
        """
        sampled_points = self.profile.sample_points() #(N_Loops,N_curves,2)
        sampled_points=sampled_points.reshape(-1,2) #(N,2)=(N_Loops*N_curves)
        sampled_points_transformed=point_transformation(sampled_points,
                                                        x_axis=self.sketch_plane.x_axis,\
                                                        y_axis=self.sketch_plane.y_axis,\
                                                        origin=self.sketch_plane.origin,iftranslation=True) #(N,3)
        return sampled_points_transformed

    @property
    def bbox(self):
        bbox_2d = self.profile.bbox  # (N,2)
        bbox_3d=point_transformation(bbox_2d,x_axis=self.sketch_plane.x_axis,\
                                    y_axis=self.sketch_plane.y_axis,\
                                    origin=self.sketch_plane.origin,iftranslation=True) #(N,3)
        return bbox_3d


class CADSequence(object):
    """A CAD modeling sequence, a series of extrude operations."""

    def __init__(self, extrude_seq, bbox=None):
        self.seq = extrude_seq
        #self.bbox = bbox

    @staticmethod
    def from_dict(all_stat):
        """construct CADSequence from json data"""
        seq = []
        # CAD Sequence consists of a list of Extrude Objects.
        for item in all_stat["timeline"]:
            item_index,item_id = item['index'],item['uuid']
            item_type,item_id_entity=CADSequence.get_extrusion_type(all_stat["entities"][item_index])
            if item_id!=item_id_entity:
                raise KeyError("Key Mismatch")
            if item_type== "ExtrudeFeature":
                # print(item_index)
                #item_index = CADSequence.get_extrusion_index(all_stat,item_id)
                # if not item_index:
                #     raise KeyError("Key Mismatch",item_type,item_id,item_index)
                extrude_ops = Extrude.from_dict(all_stat, item_index)
                if extrude_ops is not None:
                    seq.extend(extrude_ops)
        #bbox = CADSequence.bbox(seq)
        if len(seq)==0:
            #return None
            raise Exception("Failed: No CAD Sequence created. Probably there are unsupported curves.")
        return CADSequence(seq)

    @staticmethod
    def bbox(seq):
        allstartBbox = []
        allendBbox=[]
        for ex in seq:
            allstartBbox.append(ex.bbox[0])
            allendBbox.append(ex.bbox[1])
        cad_bbox=np.vstack([np.max(allendBbox,axis=0),np.min(allstartBbox,axis=0)])
        return cad_bbox

    @staticmethod
    def get_extrusion_index(all_stat, item_id):
        """ Get the index of the extrusion gived the id"""
        for index,tm in enumerate(all_stat['entities']):
            key_= list(tm.keys())[0]
            if tm[key_]["uuid"] == item_id:
                return index
        return None

    @staticmethod
    def get_extrusion_type(item):
        """
        Returns the extrusion type
        """
        key_ = list(item.keys())[0]
        try:
           return (item[key_]['type'],item[key_]['uuid'])
        except:
            return (key_,item[key_]['uuid'])

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        commands = vec[:, 0]
        ext_indices = [-1] + np.where(commands == EXT_IDX)[0].tolist()
        ext_seq = []
        for i in range(len(ext_indices) - 1):
            start, end = ext_indices[i], ext_indices[i + 1]
            ext_seq.append(Extrude.from_vector(
                vec[start+1:end+1], is_numerical, n))
        cad_seq = CADSequence(ext_seq)
        return cad_seq

    def __str__(self):
        res = ""
        for i, ext in enumerate(self.seq):
            res += "({})".format(i) + str(ext) + "\n"
        return res

    def to_vector(self, max_n_ext=10, max_n_loops=None, max_len_loop=None, max_total_len=60, pad=False):
        # if len(self.seq) > max_n_ext:
        #     return None
        vec_seq = []
        for item in self.seq:
            vec = item.to_vector(max_n_loops, max_len_loop, pad=False)
            if vec is None:
                return None
            vec = vec[:-1]  # last one is EOS, removed
            vec_seq.append(vec)

        vec_seq = np.concatenate(vec_seq, axis=0)
        vec_seq = np.concatenate([vec_seq, EOS_VEC[np.newaxis]], axis=0)

        # add EOS padding
        if pad and vec_seq.shape[0] < max_total_len:
            pad_len = max_total_len - vec_seq.shape[0]
            vec_seq = np.concatenate(
                [vec_seq, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        return vec_seq

    def transform(self, translation, scale):
        """linear transformation"""
        for item in self.seq:
            item.transform(translation, scale)

    def normalize(self, size=1.0):
        """(1)normalize the shape into unit cube (-1~1). """
        scale = size * NORM_FACTOR / np.max(np.abs(CADSequence.bbox(self.seq)))
        # Normalize the sketch profile
        for ex in self.seq:
            ex.normalize()
        #print(f"Scale {scale}")
        self.transform(0.0, scale)

    def numericalize(self, n=256):
        for item in self.seq:
            item.numericalize(n)

    def flip_sketch(self, axis):
        for item in self.seq:
            item.flip_sketch(axis)

    def random_transform(self):
        for item in self.seq:
            # random transform sketch
            scale = random.uniform(0.8, 1.2)
            item.profile.transform(-np.array([128, 128]), scale)
            translate = np.array(
                [random.randint(-5, 5), random.randint(-5, 5)], dtype=np.int32) + 128
            item.profile.transform(translate, 1)

            # random transform and scale extrusion
            t = 0.05
            translate = np.array(
                [random.uniform(-t, t), random.uniform(-t, t), random.uniform(-t, t)])
            scale = random.uniform(0.8, 1.2)
            # item.sketch_plane.transform(translate, scale)
            item.sketch_pos = (item.sketch_pos + translate) * scale
            item.extent_one *= random.uniform(0.8, 1.2)
            item.extent_two *= random.uniform(0.8, 1.2)

    def random_flip_sketch(self):
        for item in self.seq:
            flip_idx = random.randint(0, 3)
            if flip_idx > 0:
                item.flip_sketch(['x', 'y', 'xy'][flip_idx - 1])

    def sample_points(self):
        sample_points = []
        for ex in self.seq:
            sample_points.append(ex.sample_points())
        return np.concatenate(sample_points, axis=0)
