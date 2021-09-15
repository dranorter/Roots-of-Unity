import numpy as np
import itertools


class GoldenField:
    phi = 1.61803398874989484820458683

    def __init__(self, values):
        self.ndarray = np.array(values, dtype=np.int16)
        if self.ndarray.shape[-1] != 2:
            raise Exception("Not a valid golden field array; last axis must be of size 2.")

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self.ndarray)})"

    def __array__(self, dtype=None):
        return self.ndarray[..., 0] + self.phi * self.ndarray[..., 1]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            # Check if all integer
            all_integer = True
            for input in inputs:
                # if not isinstance(input ,Integral):
                if isinstance(input, np.ndarray):
                    if not (input.dtype.kind in ['u', 'i']):
                        all_integer = False
                elif isinstance(input, self.__class__):
                    pass
                else:
                    all_integer = False
            if not all_integer:
                # If we're not dealing with integers, there's no point in
                # staying a GoldenField.
                return ufunc(np.array(self), *inputs, **kwargs)

            if ufunc == np.add:
                returnval = np.zeros(self.ndarray.shape)
                returnval = returnval + self.ndarray
                for input in inputs:
                    if isinstance(input, self.__class__):
                        returnval = returnval + input.ndarray
                    else:
                        # Just add to the integer part
                        returnval[..., 0] = returnval[..., 0] + input
                return self.__class__(returnval)
            elif ufunc == np.multiply:
                returnval = self.ndarray.copy()
                for input in inputs:
                    intpart = np.zeros(self.ndarray[..., 0].shape)
                    phipart = np.zeros(self.ndarray[..., 0].shape)
                    if isinstance(input, self.__class__):
                        intpart = returnval[..., 0] * input.ndarray[..., 0]
                        phipart = returnval[..., 0] * input.ndarray[..., 1] + returnval[..., 1] * input.ndarray[..., 0]
                        intpart = intpart + returnval[..., 1] * input.ndarray[..., 1]
                        phipart = phipart + returnval[..., 1] * input.ndarray[..., 1]
                    elif isinstance(input, np.ndarray):
                        # Multiply both parts by the array
                        intpart = returnval[..., 0] * input
                        phipart = returnval[..., 1] * input
                    # elif isinstance(input, numbers.Integral):
                    #     intpart = returnval[... ,0] * input
                    #     phipart = returnval[... ,1] * input
                    else:
                        return NotImplemented
                    returnval[..., 0] = intpart
                    returnval[..., 1] = phipart
                return self.__class__(returnval)
            else:
                return NotImplemented
        else:
            return NotImplemented


base = np.zeros((3, 6))
base += 1
zerod_parts = []
for i in range(1, 6):
    new_part = np.copy(base)
    new_part[0, 0] = 0
    new_part[0, i] = 0
    for j in range(1, 4):
        np_copy = np.copy(new_part)
        subarray = [0, 1, 1, 1]
        subarray[j] = 0
        np_copy[1][np_copy[0] != 0] = subarray
        np_copy[2][np_copy[0] * np_copy[1] != 0] = 0
        zerod_parts.append(np_copy)

sign_arrays = np.array(
    [[1, 1, 1, 1], [1, -1, 1, 1], [1, 1, -1, 1], [1, -1, -1, 1], [1, 1, 1, -1], [1, -1, 1, -1], [1, 1, -1, -1],
     [1, -1, -1, -1]])
signed = []
for s in sign_arrays:
    for shape in zerod_parts:
        new_1 = np.copy(shape)
        new_1[0, new_1[0] == 1] = s
        sign_arrays2 = np.array([[1, 1], [1, -1]])
        for s_2 in sign_arrays2:
            new_2 = np.copy(new_1)
            new_2[1, new_2[0] == 0] = s_2
            sign_arrays3 = np.array([[1, -1], [-1, 1]])
            for s_3 in sign_arrays3:
                new_3 = np.copy(new_2)
                new_3[1, new_3[2] == 0] = new_3[0, new_3[0] * new_3[1] != 0] * np.array(s_3)
                new_3[2, new_3[0] == 0] = new_3[1, (new_3[0] == 0) * (new_3[1] != 0)] * np.array([1, -1])
                for s_4 in sign_arrays3:
                    new_4 = np.copy(new_3)
                    new_4[2, new_4[1] == 0] = new_4[0, new_4[1] == 0] * np.array(s_4)
                    signed.append(new_4)

phid = []
for signs in signed:
    new_phid = np.copy(signs)
    new_phid[1, new_phid[0] == 0] = new_phid[1, new_phid[0] == 0] * 1.61803398874989484820458683
    new_phid[2, new_phid[1] == 0] = new_phid[2, new_phid[1] == 0] * 1.61803398874989484820458683
    new_phid[0, new_phid[2] == 0] = new_phid[0, new_phid[2] == 0] * 1.61803398874989484820458683
    phid.append(new_phid)

for signs in signed:
    new_phid = np.copy(signs)
    new_phid[1, new_phid[2] == 0] = new_phid[1, new_phid[2] == 0] * 1.61803398874989484820458683
    new_phid[2, new_phid[0] == 0] = new_phid[2, new_phid[0] == 0] * 1.61803398874989484820458683
    new_phid[0, new_phid[1] == 0] = new_phid[0, new_phid[1] == 0] * 1.61803398874989484820458683
    phid.append(new_phid)

phid_copy = [np.copy(x) for x in phid]

orientations = [phid[0]]
for i in phid:
    skip = False
    for j in orientations:
        if not skip and np.all([np.linalg.matrix_rank([j[0], j[1], j[2], i[k]]) != 4 for k in [0, 1, 2]]):
            skip = True
    if not skip:
        orientations.append(i)

phid_map = np.zeros(len(phid))
for i in range(len(phid)):
    for j in range(len(orientations)):
        if np.all([np.linalg.matrix_rank([orientations[j][0], orientations[j][1], orientations[j][2], phid[i][k]]) != 4
                   for k in [0, 1, 2]]):
            phid_map[i] = j
            break

intersection_counts = [6 - np.linalg.matrix_rank(np.concatenate([orientations[0], orientations[i]])) for i in
                       range(len(orientations))]

line_intersections = np.array(orientations)[np.array(intersection_counts) == 1]
line_interactions = [6 - np.linalg.matrix_rank(np.concatenate([line_intersections[i], line_intersections[j]])) for i in
                     range(131) for j in range(i)]

shared_lines = [
    6 - np.linalg.matrix_rank(np.concatenate([orientations[0], line_intersections[i], line_intersections[j]])) for i in
    range(131) for j in range(i)]

intersection_matrix = np.array(
    [[6 - np.linalg.matrix_rank(np.concatenate([orientations[j], orientations[i]])) for i in range(len(orientations))]
     for j in range(len(orientations))])
point_intersections = np.nonzero(intersection_matrix[0] == 0)[0]
line_intersections = np.nonzero(intersection_matrix[0] == 1)[0]
plane_intersections = np.nonzero(intersection_matrix[0] == 2)[0]

rank_5_example = np.concatenate([orientations[0], orientations[380][:2]])
orthogonal = np.nonzero(np.all(np.array(orientations).dot(orientations[0].T) == 0, axis=(1, 2)))[0][
    0]  # orientations[317]


def get_shared_line(i, j):
    U, s, Vh = np.linalg.svd(np.concatenate([orientations[i], orientations[j]]).T)
    shared_line = orientations[i].T.dot(Vh[np.abs(s).argmin()][:3])
    shared_line[np.abs(shared_line) < 1e-15] = 0
    return shared_line


all_shared_lines = []
for i in range(len(orientations)):
    for j in range(i):
        if intersection_matrix[i, j] == 1:
            all_shared_lines.append(get_shared_line(i, j))

all_shared_lines = np.array(all_shared_lines)

u_lines = np.unique(all_shared_lines, axis=0)

unique_lines = []

intish_lines = [l / np.max(np.abs(l)) for l in u_lines]

unique_lines = []
for line in intish_lines[:100]:
    unq = True
    for uline in unique_lines:
        diff = np.linalg.norm(line - uline)
        if diff < 1e-13:
            unq = False
            break
    if unq:
        unique_lines.append(line)

for line in intish_lines[:1000]:
    unq = True
    for uline in unique_lines:
        diff = np.linalg.norm(line - uline)
        if diff < 1e-13:
            unq = False
            break
    if unq:
        unique_lines.append(line)

for line in intish_lines[1000:10000]:
    unq = True
    for uline in unique_lines:
        diff = np.linalg.norm(line - uline)
        if diff < 1e-13:
            unq = False
            break
    if unq:
        unique_lines.append(line)

unique_two = []
for line in unique_lines:
    unq = True
    for uline in unique_two:
        diff = np.linalg.norm(line - uline)
        neg_diff = np.linalg.norm(line + uline)
        if diff < 1e-13 or neg_diff < 1e-13:
            unq = False
            break
    if unq:
        unique_two.append(line)

temp = [np.count_nonzero([np.linalg.norm(orientations[o].dot(unique_two[i].T).dot(orientations[o]) / np.linalg.norm(
    orientations[o].dot(unique_two[i].T).dot(orientations[o])) - (
                                                     unique_two[i] / np.linalg.norm(unique_two[i]))) > 1e-13 for i in
                          range(len(unique_two))]) for o in range(len(orientations))]
supposedly_linecounts = [3076 - x for x in temp]

norm_worldplane_project = [orientations[i] / np.linalg.norm(orientations[i][0]) for i in range(len(orientations))]
norm_6space_coords = [orientations[i].T / np.linalg.norm(orientations[i][0]) for i in range(len(orientations))]
kalix_transformations_6D = [
    [norm_6space_coords[i].dot(norm_worldplane_project[i].dot(norm_6space_coords[j].dot(norm_worldplane_project[i])))
     for j in range(len(orientations))] for i in range(len(orientations))]
kalix_transformations_3D = [[norm_worldplane_project[i].dot(norm_6space_coords[j]) for j in range(len(orientations))]
                            for i in range(len(orientations))]
kalix_angles = [np.linalg.svd(kalix_transformations_3D[0][i])[1] for i in range(len(orientations))]

unique_kalix_angles = []
for angle in kalix_angles:
    unique = True
    for u_angle in unique_kalix_angles:
        diff = angle - u_angle
        if np.linalg.norm(diff) < 1e-15:
            unique = False
            break
    if unique:
        unique_kalix_angles.append(angle)

unique_kalix_angles_map = np.zeros((len(kalix_angles),), dtype=np.int)
for i in range(len(kalix_angles)):
    for j in range(len(unique_kalix_angles)):
        diff = kalix_angles[i] - unique_kalix_angles[j]
        if np.linalg.norm(diff) < 1e-15:
            unique_kalix_angles_map[i] = j
            break

kalix_angles_6D = [np.linalg.svd(kalix_transformations_6D[0][i])[1] for i in range(len(orientations))]

unique_kalix_angles_6D = []
for angle in kalix_angles_6D:
    unique = True
    for u_angle in unique_kalix_angles_6D:
        diff = angle - u_angle
        if np.linalg.norm(diff) < 1e-15:
            unique = False
            break
    if unique:
        unique_kalix_angles_6D.append(angle)

discrepencies = [
    intersection_matrix[0][list(unique_kalix_angles_map).index(unique_kalix_angles_map[i])] != intersection_matrix[0][i]
    for i in range(len(unique_kalix_angles_map))]

# Now I want to think of the 6D angles in a different way.
# Essentially, I'd like to rotate the basis of each orientation so that they're standardized slightly differently.
# [1,0,0,0,0,0] will always point the same direction (arbitrarily, [0, phi, 1]). [0,1,0,0,0,0] will point one of two
# collinear directions (either [0, -phi, 1] or [0, phi, -1]). [0,0,1,0,0,0] will be limited to a certain plane, such
# that we don't get any basis which is just a reflection of some other basis. (IE, the plane of restriction is not
# symmetrical about the plane formed by the first two basis elements' projection.) The remaining basis vectors can
# project anywhere that produces the icosahedral symmetry.

phi = 1.61803398874989484820458683
standard_basis = np.array([[phi, 0., 1., phi, 0., -1.],
                           [1., phi, 0., -1., phi, -0.],
                           [0., 1., phi, 0., -1., phi]])

orientations2 = [standard_basis]
orientations2.append(standard_basis * np.array([1, -1, 1, 1, 1, 1]))
orientations2 = [orntn.T[arrangement] for orntn in orientations2 for arrangement in
                 [[0, 1] + list(perm) for perm in itertools.permutations([2, 3, 4, 5]) if perm[0] in [2, 3]]]
orientations2 = [orntn.T * np.array(signage) for orntn in orientations2 for signage in
                 [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, -1], [1, 1, 1, 1, -1, 1], [1, 1, 1, 1, -1, -1],
                  [1, 1, 1, -1, 1, 1], [1, 1, 1, -1, 1, -1], [1, 1, 1, -1, -1, 1], [1, 1, 1, -1, -1, -1],
                  [1, 1, -1, 1, 1, 1], [1, 1, -1, 1, 1, -1], [1, 1, -1, 1, -1, 1], [1, 1, -1, 1, -1, -1],
                  [1, 1, -1, -1, 1, 1], [1, 1, -1, -1, 1, -1], [1, 1, -1, -1, -1, 1], [1, 1, -1, -1, -1, -1]]]

# The strategy in orientations2 was to filter out half the permutations so that we don't have chiral copies of the same
# bases. However, this makes them stop working as a group. Orientations3 singles out one canonical form while preserving
# the full list.

perm_set_chiral = [[0, 1] + list(perm) for perm in itertools.permutations([2, 3, 4, 5]) if list(perm).index(2) in [0, 1]]
perm_set_sinestral = [[0, 1] + list(perm) for perm in itertools.permutations([2, 3, 4, 5]) if list(perm).index(2) not in [0, 1]]
#perm_set = [[0, 1] + list(perm) for perm in itertools.permutations([2, 3, 4, 5])]
perm_set = perm_set_chiral + perm_set_sinestral
sign_set = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, -1], [1, 1, 1, 1, -1, 1], [1, 1, 1, 1, -1, -1], [1, 1, 1, -1, 1, 1],
            [1, 1, 1, -1, 1, -1], [1, 1, 1, -1, -1, 1], [1, 1, 1, -1, -1, -1], [1, 1, -1, 1, 1, 1],
            [1, 1, -1, 1, 1, -1], [1, 1, -1, 1, -1, 1], [1, 1, -1, 1, -1, -1], [1, 1, -1, -1, 1, 1],
            [1, 1, -1, -1, 1, -1], [1, 1, -1, -1, -1, 1], [1, 1, -1, -1, -1, -1]]
sign_set = [np.array(s) * np.array(s2) for s in sign_set for s2 in [[1, 1, 1, 1, 1, 1], [1, -1, 1, 1, 1, 1]]]
permutations_chiral = [(signage, permage) for signage in sign_set for permage in perm_set_chiral]
permutations = [(signage, permage) for signage in sign_set for permage in perm_set_chiral] + [(signage, permage) for signage in sign_set for permage in perm_set_sinestral]


def apply_backwards(o, p):
    return (o * p[0]).T[p[1]].T

def apply(o, p):
    product = (o * p[0])
    result = np.zeros_like(o)
    result.T[p[1]] = product.T
    return result


orientations3 = [apply(standard_basis, p) for p in permutations_chiral]
orientations3_achiral = [apply(standard_basis, p) for p in permutations]
# Useful lookup table for converting to canonical, but takes awhile to calculate
orientations3_chiral = [np.nonzero([6-np.linalg.matrix_rank(np.concatenate([achiral,x])) == 3 for x in orientations3])[0][0] for achiral in orientations3_achiral]

def apply_list_backwards(o, l):
    # Takes indices into the permutations list
    result = o
    for i in l:
        result = apply_backwards(result, permutations[i])
    # We return the orientations3 index instead of the result
    matches = np.nonzero(np.all(np.array(orientations3) - result == 0, axis=(1, 2)))[0]
    if len(matches) > 0:
        return matches[0]
    else:
        # Needs flipped
        o2 = apply_backwards(o, (np.array([1, 1, -1, 1, 1, -1]), [0, 1, 5, 4, 3, 2]))
        result = o2
        for i in l:
            result = apply_backwards(result, permutations[i])
        # We return the orientations3 index instead of the result
        matches = np.nonzero(np.all(np.array(orientations3) - result == 0, axis=(1, 2)))[0]
        if len(matches) > 0:
            return matches[0]

def apply_list(o, l):
    # Takes indices into the permutations list
    result = o
    for i in l:
        result = apply(result, permutations[i])
    # We return the index instead of the result. Note that it is achiral and may need to be converted if the
    # canonical form is needed.
    matches = np.nonzero(np.all(np.array(orientations3_achiral) - result == 0, axis=(1, 2)))[0]
    if len(matches) > 0:
        return matches[0]

ortho_spaces = [np.nonzero(np.all(np.array(orientations3).dot(orientations3[i].T) == 0, axis=(1, 2)))[0][0] for i in
                range(len(orientations3))]

inverse_perms = [apply_list(orientations3_achiral[i],[i]*([apply_list(orientations3_achiral[i],[i]*n) for n in range(20)].index(0) - 1)) for i in range(768)]

intersection_matrix3 = np.array([[6 - np.linalg.matrix_rank(np.concatenate([orientations3[j], orientations3[i]])) for i
                                  in range(len(orientations3))] for j in range(len(orientations3))])

clean_cycles = np.array([intersector for x, intersector in [
    (intersection_matrix3[0, np.array(orientations3_chiral)[[apply_list(orientations3[0], [intersector] * x) for x in range(9)]]], intersector) for
    intersector in np.nonzero(intersection_matrix3[0] == 0)[0]] if
                         x[-1] == 3 and list(x).count(3) == 2 and list(x).count(0) == 6])

clean_cycles_extended = np.array([intersector for x, intersector in [
    ([6 - np.linalg.matrix_rank(np.concatenate([standard_basis, orientations3_achiral[y]])) for y in
     [apply_list(orientations3[0], [intersector] * x) for x in range(9)]], intersector) for
    intersector in range(768)] if
                         x[-1] == 3 and list(x).count(3) == 2 and list(x).count(0) == 6])

kalix_dozen = [[[1, 1, 1, 1, -1, 1], [0, 1, 2, 5, 4, 3]],
               [[1, 1, 1, -1, 1, 1], [0, 1, 2, 3, 4, 5]],
               [[1, 1, 1, 1, 1, 1], [0, 1, 3, 2, 5, 4]],
               [[1, 1, 1, -1, -1, -1], [0, 1, 3, 4, 2, 5]],
               [[1, 1, -1, -1, -1, -1], [0, 1, 2, 5, 3, 4]],
               [[1, 1, -1, -1, -1, -1], [0, 1, 4, 2, 5, 3]],
               [[1, 1, 1, 1, -1, 1], [0, 2, 4, 3, 5, 1]],#
               [[1, 1, 1, 1, -1, -1], [0, 2, 4, 5, 1, 3]],#
               [[1, 1, 1, -1, -1, 1], [0, 2, 5, 1, 4, 3]],#
               [[1, 1, 1, -1, -1, -1], [0, 3, 5, 1, 2, 4]],#
               [[1, 1, -1, -1, -1, 1], [0, 4, 1, 5, 3, 2]],#
               [[1, -1, -1, -1, -1, 1], [0, 1, 2, 3, 4, 5]]]

kalix_dozen_indices = [apply_list(apply(standard_basis, p),[]) for p in kalix_dozen]
replacement_k_indices = [x for x in range(384) if np.all(intersection_matrix3[x,[orientations3_chiral[k] for k in kalix_dozen_indices if k is not None]] == 0)]
# haven't needed to check that the replacement indices intersect one another in points
for i in range(kalix_dozen_indices.count(None)):
    kalix_dozen_indices[kalix_dozen_indices.index(None)] = replacement_k_indices[i]
dozen_at_origin = [apply_list(orientations3_achiral[x], [inverse_perms[kalix_dozen_indices[0]]]) for x in kalix_dozen_indices]

def get_shared_line(i, j):
    U, s, Vh = np.linalg.svd(np.concatenate([orientations3_achiral[i], orientations3_achiral[j]]).T)
    shared_line = orientations3_achiral[i].T.dot(Vh[np.abs(s).argmin()][:3])
    shared_line[np.abs(shared_line) < 1e-15] = 0
    return shared_line

proposed_set = [0,ortho_spaces[0]]
for o in range(len(orientations3)):
    if np.all(intersection_matrix[o,proposed_set] == 0):
        proposed_set.append(o)

# This set doesn't make a good collection of world-spaces since it intersects in lines and planes so frequently, but
# conceptually it's a nice set because it's just different combinations of 3 vectors from the overworld and
# the three vectors from ⊥. (Sadly not simply the three basis vectors of each - the 18 extra spaces resulting from
# that don't preserve the symmetry needed for the aperiodic grid.)
cardinal_orientations = ([0,ortho_spaces[0]] +
        list(np.nonzero(intersection_matrix3[0] == 2)[0][np.nonzero(intersection_matrix3[ortho_spaces[0],np.nonzero(intersection_matrix3[0] == 2)[0]])[0]]) +
        list(np.nonzero(intersection_matrix3[ortho_spaces[0]] == 2)[0][np.nonzero(intersection_matrix3[0,np.nonzero(intersection_matrix3[ortho_spaces[0]] == 2)[0]])[0]])
)

# Here is a "clean cycle" generator which creates its one shared line in a cardinal direction (namely z, ie,
# standard_basis[2]). It's not part of the canonical set of permutations.
cardinal_cycle = ([-1,-1,1,1,1,1],[0,4,1,3,5,2])
cardinal_cycle_plane = [6-np.linalg.matrix_rank(np.concatenate([orientations3[x],temp[4]])) for x in range(384)].index(3)

x_lines = [x for x in range(381) if intersection_matrix3[0,x] == 1 and
           np.linalg.norm(np.abs(standard_basis[0].dot(get_shared_line(0,x)))
                                                   - np.linalg.norm(standard_basis[0]*np.linalg.norm(get_shared_line(0,x)))) < 1e-13]
y_lines = [x for x in range(381) if intersection_matrix3[0,x] == 1 and
           np.linalg.norm(np.abs(standard_basis[1].dot(get_shared_line(0,x)))
                                                   - np.linalg.norm(standard_basis[1]*np.linalg.norm(get_shared_line(0,x)))) < 1e-13]
z_lines = [x for x in range(381) if intersection_matrix3[0,x] == 1 and
           np.linalg.norm(np.abs(standard_basis[2].dot(get_shared_line(0,x)))
                                                   - np.linalg.norm(standard_basis[2]*np.linalg.norm(get_shared_line(0,x)))) < 1e-13]