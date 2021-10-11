PRO estimate_z,lib_mag,input_mag,lib_z,z_estimate

; This procedure provides an estimate of the Z axis location of the object by comparing
; the magnitudes of the NFD's of the input contour and the library contour, with a known
; z location for the library contour.

	COMMON CALIBRATION,scale,principle_distance,pers_dist,x_off,y_off

	z_estimate = principle_distance - (lib_mag/input_mag)*(principle_distance-lib_z)

	end