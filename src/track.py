from typing import Self, overload
import numpy as np;
import mediapipe as mp

# a data type as a storage for physical quantity measurement of each body part
# data could have dimension which is the dimension of the tensor
# e.g. a dimension of 1 will store a scalar in each body part
#.     a dimension of (1, 3) will sotre a vector in each body part
class BodyData:
    
    def __init__(self, dimension):
        self.dimension = dimension
        self.head = np.zeros(dimension)
        self.body = np.zeros(dimension)
        self.right_shoulder = np.zeros(dimension)
        self.right_arm = np.zeros(dimension)
        self.right_hand = np.zeros(dimension)
        self.right_thumb = np.zeros(dimension)
        self.left_shoulder = np.zeros(dimension)
        self.left_arm = np.zeros(dimension)
        self.left_hand = np.zeros(dimension)
        self.left_thumb = np.zeros(dimension)
        self.right_thigh = np.zeros(dimension)
        self.right_leg = np.zeros(dimension)
        self.right_feet = np.zeros(dimension)
        self.left_thigh = np.zeros(dimension)
        self.left_leg = np.zeros(dimension)
        self.left_feet = np.zeros(dimension)

    def __add__(self, o: Self) -> Self:
        result = BodyData(self.dimension)
        result.head = self.head + o.head
        result.body = self.body + o.body
        result.right_shoulder = self.right_shoulder + o.right_shoulder
        result.right_arm = self.right_arm + o.right_arm
        result.right_hand = self.right_hand + o.right_hand
        result.right_thumb = self.right_thumb + o.right_thumb
        result.left_shoulder = self.left_shoulder + o.left_shoulder
        result.left_arm = self.left_arm + o.left_arm
        result.left_hand = self.left_hand + o.left_hand
        result.left_thumb = self.left_thumb + o.left_thumb
        result.right_thigh = self.right_thigh + o.right_thigh
        result.right_leg = self.right_leg + o.right_leg
        result.right_feet = self.right_feet + o.right_feet
        result.left_thigh = self.left_thigh + o.left_thigh
        result.left_leg = self.left_leg + o.left_leg
        result.left_feet = self.left_feet + o.left_feet

        return result

    @overload
    def __mul__(self, o: Self) -> Self: ...

    @overload
    def __mul__(self, o: float) -> Self: ...

    def __mul__(self, o: object) -> Self:
        if (isinstance(o, BodyData)):
            result = BodyData(self.dimension)
            result.head = self.head * o.head
            result.body = self.body * o.body
            result.right_shoulder = self.right_shoulder * o.right_shoulder
            result.right_arm = self.right_arm * o.right_arm
            result.right_hand = self.right_hand * o.right_hand
            result.right_thumb = self.right_thumb * o.right_thumb
            result.left_shoulder = self.left_shoulder * o.left_shoulder
            result.left_arm = self.left_arm * o.left_arm
            result.left_hand = self.left_hand * o.left_hand
            result.left_thumb = self.left_thumb * o.left_thumb
            result.right_thigh = self.right_thigh * o.right_thigh
            result.right_leg = self.right_leg * o.right_leg
            result.right_feet = self.right_feet * o.right_feet
            result.left_thigh = self.left_thigh * o.left_thigh
            result.left_leg = self.left_leg * o.left_leg
            result.left_feet = self.left_feet * o.left_feet

            return result
        elif (isinstance(o, float)):
            result = BodyData(self.dimension)
            result.head = self.head * o
            result.body = self.body * o
            result.right_shoulder = self.right_shoulder * o
            result.right_arm = self.right_arm * o
            result.right_hand = self.right_hand * o
            result.right_thumb = self.right_thumb * o
            result.left_shoulder = self.left_shoulder * o
            result.left_arm = self.left_arm * o
            result.left_hand = self.left_hand * o
            result.left_thumb = self.left_thumb * o
            result.right_thigh = self.right_thigh * o
            result.right_leg = self.right_leg * o
            result.right_feet = self.right_feet * o
            result.left_thigh = self.left_thigh * o
            result.left_leg = self.left_leg * o
            result.left_feet = self.left_feet * o

            return result
        
        return NotImplemented
    
    @overload
    def __rmul__(self, o: Self) -> Self: ...
    
    @overload
    def __rmul__(self, o: float) -> Self: ...

    def __rmul__(self, o: object) -> Self:
        if (isinstance(o, BodyData)):
            return o * self
        elif (isinstance(o, float)):
            return self * o
        
        return NotImplemented

    @overload
    def __truediv__(self, o: float) -> Self: ...

    def __truediv__(self, o: object) -> Self:
        if (isinstance(o, float)):
            return self.__mul__(1.0 / o)
        
        return NotImplemented

    def __sub__(self, o: Self) -> Self:
        return self + (-1.0 * o)

    # returns the magnitude of all body data
    def get_mag(self):
        result = BodyData(1)
        result.head = np.linalg.norm(self.head)
        result.body = np.linalg.norm(self.body)
        result.right_shoulder = np.linalg.norm(self.right_shoulder)
        result.right_arm = np.linalg.norm(self.right_arm)
        result.right_hand = np.linalg.norm(self.right_hand)
        result.right_thumb = np.linalg.norm(self.right_thumb)
        result.right_thigh = np.linalg.norm(self.right_thigh)
        result.right_leg = np.linalg.norm(self.right_leg)
        result.right_feet = np.linalg.norm(self.right_feet)
        result.left_shoulder = np.linalg.norm(self.left_shoulder)
        result.left_arm = np.linalg.norm(self.left_arm)
        result.left_hand = np.linalg.norm(self.left_hand)
        result.left_thumb = np.linalg.norm(self.left_thumb)
        result.left_thigh = np.linalg.norm(self.left_thigh)
        result.left_leg = np.linalg.norm(self.left_leg)
        result.left_feet = np.linalg.norm(self.left_feet)
    
        return result

    # returns the sum of all body data
    def get_net(self):
        return self.head + self.body + self.right_shoulder + self.right_arm + self.right_hand + self.right_thumb + self.left_shoulder + self.left_arm + self.left_hand + self.left_thumb + self.right_thigh + self.right_leg + self.right_feet + self.left_thigh + self.left_leg + self.left_feet
    
    # returns the magnitude of the sum of all body data
    def get_net_mag(self):
        return np.linalg.norm(self.get_net())
    
    def __str__(self):
        return f"(\n    Head: {self.head},\n    Body: {self.body},\n    Right Arm: {self.right_arm},\n    Left Arm: {self.left_arm},\n    Right Leg: {self.right_leg},\n    Left Leg: {self.left_leg}\n)"
    
    def copy(self, o: Self):
        self.head = o.head
        self.body = o.body
        self.right_shoulder = o.right_shoulder
        self.right_arm = o.right_arm
        self.right_hand = o.right_hand
        self.right_thumb = o.right_thumb
        self.right_thigh = o.right_thigh
        self.right_leg = o.right_leg
        self.right_feet = o.right_feet

        self.left_shoulder = o.left_shoulder
        self.left_arm = o.left_arm
        self.left_hand = o.left_hand
        self.left_thumb = o.left_thumb
        self.left_thigh = o.left_thigh
        self.left_leg = o.left_leg
        self.left_feet = o.left_feet

# takes in the measurement of the position of each body joint and convert it to
# center of mass position of each body parts
# use the center of mass position of each body parts to compute the net momemtum of each body parts
class BodyTracker:

    def __init__(self, total_mass, gender="male"):
        self.visibility = BodyData(1)
        self.pos = BodyData(3)
        self.prev_pos = BodyData(3)
        self.vel = BodyData(3)
        self.prev_vel = BodyData(3)
        self.mass = BodyData(1)
        self.momemtum = BodyData(3)
        self.prev_momemtum = BodyData(3)
        self.force = BodyData(3)

        self.male_mass_proportion = BodyData(1)
        self.female_mass_proportion = BodyData(1)
        
        self.setup_mass_proportion()
        self.setup_mass(total_mass, gender)

    # loads the value of male mass proportion and female mass proportion
    # using reference to https://robslink.com/SAS/democd79/body_part_weights.htm
    def setup_mass_proportion(self):
        self.male_mass_proportion.head = 0.0826
        self.male_mass_proportion.body = 0.469
        self.male_mass_proportion.left_shoulder = 0.0325
        self.male_mass_proportion.left_arm = 0.0187
        self.male_mass_proportion.left_hand = 0.0065
        self.male_mass_proportion.left_thumb = 0.001 #guessed
        self.male_mass_proportion.left_thigh = 0.105
        self.male_mass_proportion.left_leg = 0.0475
        self.male_mass_proportion.left_feet = 0.0143
        self.male_mass_proportion.right_shoulder = 0.0325
        self.male_mass_proportion.right_arm = 0.0187
        self.male_mass_proportion.right_hand = 0.0065
        self.male_mass_proportion.right_thumb = 0.001
        self.male_mass_proportion.right_thigh = 0.105
        self.male_mass_proportion.right_leg = 0.0475
        self.male_mass_proportion.right_feet = 0.0143

        self.female_mass_proportion.head = 0.0820
        self.female_mass_proportion.body = 0.452
        self.female_mass_proportion.left_shoulder = 0.0290
        self.female_mass_proportion.left_arm = 0.0157
        self.female_mass_proportion.left_hand = 0.005
        self.female_mass_proportion.left_thumb = 0.001 #guessed
        self.female_mass_proportion.left_thigh = 0.118
        self.female_mass_proportion.left_leg = 0.0535
        self.female_mass_proportion.left_feet = 0.0133
        self.female_mass_proportion.right_shoulder = 0.0290
        self.female_mass_proportion.right_arm = 0.0157
        self.female_mass_proportion.right_hand = 0.005
        self.female_mass_proportion.right_thumb = 0.001
        self.female_mass_proportion.right_thigh = 0.118
        self.female_mass_proportion.right_leg = 0.0535
        self.female_mass_proportion.right_feet = 0.0133

    # setup the mass of the body by the total mass and gender
    def setup_mass(self, total_mass, gender="male"):
        if (gender == "male"):
            self.mass = total_mass * self.male_mass_proportion
        elif (gender == "female"):
            self.mass = total_mass * self.female_mass_proportion
        else:
            return NotImplemented


    # update the position of the body
    def read_pos(self, world_landmarks):
        np_world_landmarks = []

        for i in range(len(world_landmarks)):
            np_world_landmarks.append(np.array([world_landmarks[i].x, world_landmarks[i].y, world_landmarks[i].z]))

        self.prev_pos.copy(self.pos)

        self.pos.body = 0.25 * (np_world_landmarks[12] + np_world_landmarks[11] + np_world_landmarks[24] + np_world_landmarks[23])
        self.pos.head = np_world_landmarks[0]
        self.pos.left_shoulder = 0.5 * (np_world_landmarks[11] + np_world_landmarks[13])
        self.pos.left_arm = 0.5 * (np_world_landmarks[13] + np_world_landmarks[15])
        self.pos.left_hand = 0.333 * (np_world_landmarks[15] + np_world_landmarks[19] + np_world_landmarks[17])
        self.pos.left_thumb = 0.5 * (np_world_landmarks[15] + np_world_landmarks[21])
        self.pos.left_thigh = 0.5 * (np_world_landmarks[23] + np_world_landmarks[25])
        self.pos.left_leg = 0.5 * (np_world_landmarks[25] + np_world_landmarks[27])
        self.pos.left_feet = 0.333 * (np_world_landmarks[27] + np_world_landmarks[29] + np_world_landmarks[31])
        self.pos.right_shoulder = 0.5 * (np_world_landmarks[12] + np_world_landmarks[14])
        self.pos.right_arm = 0.5 * (np_world_landmarks[14] + np_world_landmarks[16])
        self.pos.right_hand = 0.333 * (np_world_landmarks[16] + np_world_landmarks[18] + np_world_landmarks[20])
        self.pos.right_thumb = 0.5 * (np_world_landmarks[16] + np_world_landmarks[22])
        self.pos.right_thigh = 0.5 * (np_world_landmarks[24] + np_world_landmarks[26])
        self.pos.right_leg = 0.5 * (np_world_landmarks[26] + np_world_landmarks[28])
        self.pos.right_feet = 0.333 * (np_world_landmarks[28] + np_world_landmarks[30] + np_world_landmarks[32])

    # update the visibility of each body parts
    def read_visibility(self, world_landmarks):
        self.visibility.body = 0.25 * (world_landmarks[12].visibility + world_landmarks[11].visibility + world_landmarks[24].visibility + world_landmarks[23].visibility)
        self.visibility.head = world_landmarks[0].visibility
        self.visibility.left_shoulder = 0.5 * (world_landmarks[11].visibility + world_landmarks[13].visibility)
        self.visibility.left_arm = 0.5 * (world_landmarks[13].visibility + world_landmarks[15].visibility)
        self.visibility.left_hand = 0.333 * (world_landmarks[15].visibility + world_landmarks[19].visibility + world_landmarks[17].visibility)
        self.visibility.left_thumb = 0.5 * (world_landmarks[15].visibility + world_landmarks[21].visibility)
        self.visibility.left_thigh = 0.5 * (world_landmarks[23].visibility + world_landmarks[25].visibility)
        self.visibility.left_leg = 0.5 * (world_landmarks[25].visibility + world_landmarks[27].visibility)
        self.visibility.left_feet = 0.333 * (world_landmarks[27].visibility + world_landmarks[29].visibility + world_landmarks[31].visibility)
        self.visibility.right_shoulder = 0.5 * (world_landmarks[12].visibility + world_landmarks[14].visibility)
        self.visibility.right_arm = 0.5 * (world_landmarks[14].visibility + world_landmarks[16].visibility)
        self.visibility.right_hand = 0.333 * (world_landmarks[16].visibility + world_landmarks[18].visibility + world_landmarks[20].visibility)
        self.visibility.right_thumb = 0.5 * (world_landmarks[16].visibility + world_landmarks[22].visibility)
        self.visibility.right_thigh = 0.5 * (world_landmarks[24].visibility + world_landmarks[26].visibility)
        self.visibility.right_leg = 0.5 * (world_landmarks[26].visibility + world_landmarks[28].visibility)
        self.visibility.right_feet = 0.333 * (world_landmarks[28].visibility + world_landmarks[30].visibility + world_landmarks[32].visibility)

    # updates all physical quantity of the body
    def update(self, world_landmarks, delta_time):

        self.read_pos(world_landmarks)
        self.read_visibility(world_landmarks)

        self.prev_vel = self.vel
        self.vel = (self.pos - self.prev_pos) / delta_time

        self.prev_momemtum = self.momemtum
        self.momemtum = self.mass * self.vel

        self.force = ((self.momemtum - self.prev_momemtum) / delta_time) * self.visibility
    



