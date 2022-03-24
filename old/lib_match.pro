FUNCTION lib_match,n_neighbors,y_pic_offset

; This procedure does library matching of the current object outline with the
; current library. It finds the n_neighbors closest matches and then interpolates
; these contours to most closely match the current object outline. In addition,
; the angles at which the nearest neighbors were generated are interpolated to 
; give an estimate of the x and y rotations of the current outline.

	COMMON ORIENT,rx_val,ry_val,rz_val,drx,dry,drz,coord_sys,cumulative
	COMMON POSITION,tx_val,ty_val,tz_val
	COMMON SAMPLING_VALUES,nsamp,samp_val,nsamp_old,win_val,win_val_old
	COMMON NFD_VARS,han_win,x_trunc,y_trunc,real,img,k,nfd_dc, $
			nfd_mag,nfd_angle
	COMMON LIBRARY,surface,dc,mag,lib_angle,k_vals,num_k,rxinc,ryinc
	COMMON PC_LIB,num_pc,num_pc_vars,prin_comp,eig_vect
	COMMON LIMITS,txmax,txmin,tymax,tymin,tzmax,tzmin,rxmax, $ 
		      rxmin,rymax,rymin,rzmax,rzmin
	COMMON CALIBRATION,scale,principle_distance,pers_dist,x_off,y_off
	COMMON ESTIMATES,x_angle_corr,y_angle_corr,z_angle,x_est,y_est,z_est

;	ON_ERROR,3
	if (N_PARAMS() eq 0) then n_neighbors = 4
	if (N_PARAMS() eq 1) then y_pic_offset = 256

	print,'Normalization coefficient of contour is =',k
; First find closest matches within library ****************************

	siz=size(mag)	; Give distance array correct dimension and type
	dist = replicate(10000.0,siz(1),siz(2))
	input = complex(real,img)

	k_index = where(k_vals eq k(0)) & k_index = k_index(0)
	if (k_index(0) eq -1) then begin
	    print,'Normalization Coefficient Not Accommodated in Present Library !!!'
	    print,'\007'
	    return,1000.0
	endif

	s = reform2(surface(k_index,*,*,*))		; Library array
	p = reform2(prin_comp(k_index,*,*,*))		; Principle component library
	pc_samp = num_pc(k_index)			; Number of principle components
	lib_ang = reform2(lib_angle(k_index,*,*))		; Normalization angle
	lo = nsamp/2 - (num_pc_vars(k_index)-1)/2
	hi = nsamp/2 + (num_pc_vars(k_index)-1)/2
	e_array = reform2(conj(eig_vect(k_index,0:num_pc_vars(k_index)-1,0:num_pc(k_index)-1)))
	input1 = transpose(e_array)#input(lo:hi) ; Xformed input

; Calculate distances between input and library (principal component) entries

	for i=0,rxinc-1 do begin
	  for j = 0,ryinc-1  do begin
	      diff1 = p(*,i,j)-input1(*) ; difference of two complex vectors
	      dist(i,j) = float(transpose(conj(diff1))#diff1) ; distance squared
	  endfor
	endfor

;	window,/free,xsize=512,ysize=512,xpos=5,ypos=500
;	temp_win = !D.WINDOW
;	t3d,/reset
;	!P.T3D = 0
;	surface,dist
;	hak,mesg='Hit any key to continue .... \007'
;	!P.T3D = 1
;	wdelete,temp_win

	order = sort(dist)
	abc = order mod rxinc	
	def = order/rxinc	

; For the four closest matches, look at the four triangular neighborhoods
; surrounding each one. If least squares interpolation results in three positive coefficients
; then proceed. If interpolation has only two positive coefficients, then redo interp. with
; those two entries. If the redone interp has two positive coefficients, then proceed, else
; result is just original library entry without interpolation. If original interpolation results
; in only one positive coefficient, then just the library entry is used. The corresponds to
; the method of Wallace and Wintz as cited in the procedure NFD.PRO.

	num_tries = 4	; Look at four closest matches from library
	alpha_array = fltarr(3,num_tries,4)
	index = [1,0,-1,0,1,0,-1,0]
	dist1 = replicate(10000.0,num_tries,4)
	for j = 0,num_tries-1 do begin		; Find best local neighborhood
	    for i=0,3 do begin			; Examine 4 triangular neighborhoods
		xindex = [abc(j),abc(j)+index(i),abc(j)+index(i+1)]
		yindex = [def(j),def(j)+index(i+3),def(j)+index(i+4)]
		xmax = max(xindex,min=xmin)
		ymax = max(yindex,min=ymin)
		if (xmax le rxinc-1 and xmin ge 0 and ymax le ryinc-1 $
		    and ymin ge 0) then begin
			array = complexarr(pc_samp,3)
			for ii=0,2 do array(*,ii) = p(0:pc_samp-1,xindex(ii),yindex(ii))
			neighbor_fit,abs(input1),abs(array),alpha
			positive = where(alpha ge 0.0 AND alpha le 1.0)
			siz = size(positive)
			if (siz(0) eq 1) then begin
			    if (siz(3) eq 3) then begin
				alpha_array(*,j,i) = alpha
				new_freq = array#alpha
				diff = new_freq(*)-input1(*);diff of 2 complex vectors
				dist1(j,i) = float(transpose(conj(diff))#diff) ; dist^2
			    endif else if (siz(3) eq 2) then begin
			        array = array(*,positive)
			        neighbor_fit,abs(input1),abs(array),alpha
			        if (alpha(0) gt 0.0 AND alpha(0) lt 1.0 AND $
				    alpha(1) gt 0.0 AND alpha(1) lt 1.0) then begin
				    alpha_array(positive,j,i) = alpha
				    new_freq = array#alpha
				    diff = new_freq(*)-input1(*)
				    dist1(j,i) = float(transpose(conj(diff))#diff) 
			        endif else begin
				    alpha_array(0,j,i) = 1.0
				    new_freq = p(*,abc(j),def(j))
				    diff = new_freq(*)-input1(*);diff of 2 complex vectors
				    dist1(j,i) = float(transpose(conj(diff))#diff) ; dist^2
				endelse				    
			    endif else if (siz(3) eq 1) then begin
				alpha_array(0,j,i) = 1.0
				new_freq = p(*,abc(j),def(j))
				diff = new_freq(*)-input1(*);diff of 2 complex vectors
				dist1(j,i) = float(transpose(conj(diff))#diff) ; dist^2
			    endif
 			endif else dist1(j,i) = 100000.0 ; No match,just big number	
		endif else begin
			alpha_array(0,j,i) = 1.0
			new_freq = p(*,abc(j),def(j))
			diff = new_freq(*)-input1(*);diff of 2 complex vectors
			dist1(j,i) = float(transpose(conj(diff))#diff) ; dist^2
			print,'Matching off edge of library, border entry used for matching'
		endelse
	    endfor
	endfor		

; Now determine the closest match - based on interpolated fits, determine which is closest
; in terms of the "Euclidean Distance" beween the interpolated principle components and the
; input principle components (from the transformed NFD's of the target shape).

    its_max = 8
    estimates = fltarr(its_max,6)
    errors = fltarr(its_max,6)
    new_e = fltarr(6)
    contours = complexarr(its_max,nsamp)
    weights = fltarr(its_max)

    order = sort(dist1)	; Sort distances for best fitting results

    print,'representative distances'
    print,format='(E20.12)',dist1(order(0:4))
    print,''

    for its = 0,its_max-1 do begin
;	print,'iterations =',its

	point = order(its) mod num_tries
	quad = order(its)/num_tries
	xindex = [abc(point),abc(point)+index(quad),abc(point)+index(quad+1)]
	yindex = [def(point),def(point)+index(quad+3),def(point)+index(quad+4)]
print,'xindex =',xindex
print,'yindex =',yindex
		
	mag_match = 0.0 & rz_match = 0.0 & tz_match = 0.0
	dc_lib = 0.0
	x_angle = 0.0 & y_angle = 0.0
	new_freq = complexarr(nsamp)
	rx = (rxmax-rxmin)/(rxinc-1)
	ry = (rymax-rymin)/(ryinc-1)
print,'rx =',rx,'	ry =',ry

	for i=0,2 do begin
	  if (xindex(i) lt 0 OR xindex(i) ge rxinc OR yindex(i) lt 0 OR yindex(i) ge ryinc) $
	    then print,'Entry off edge of library ..... ignoring component' $
	    else begin
		new_freq = new_freq + alpha_array(i,point,quad)*s(*,xindex(i),yindex(i))
		x_angle = x_angle+alpha_array(i,point,quad)*(xindex(i)*rx+rxmin)
		y_angle = y_angle+alpha_array(i,point,quad)*(yindex(i)*ry+rymin)
		mag_match = mag_match+alpha_array(i,point,quad)*mag(xindex(i),yindex(i))
		rz_match = rz_match+alpha_array(i,point,quad)*lib_ang(xindex(i),yindex(i))
		dc_lib = dc_lib + alpha_array(i,point,quad)*dc(xindex(i),yindex(i))
	    endelse
	endfor

; Estimate z rotation

	z_angle = rz_match-nfd_angle		
	zrad = z_angle*!PI/180.0		; z angle in radians

; Estimate Z translation

	tz_match = (tzmax+tzmin)/2.0
	estimate_z,mag_match,nfd_mag,tz_match,z_est	; Estimate z translation

	if (z_est eq principle_distance) then begin
	    print,'Error in calculating z position .... bombing out!'
	    print,'\007'
	    x_est = 0.0 & y_est = 0.0 & z_est = -1.0
	    x_angle_corr = 0.0 & y_angle_corr = 0.0 & z_angle = 0.0
	    return,1000
	endif

; Estimate x and y translations

	    x_trans = 256. + x_off*512./scale	; Compensate for translation of 
	    y_trans = 256. + y_off*512./scale	; principle point for zero point

	    zoom = (principle_distance-tz_match)/(principle_distance-z_est) ;zoom factor
	    x_dc = (float(dc_lib)-x_trans)*zoom	; Library x to new z and scale
	    y_dc =(imaginary(dc_lib)-y_trans)*zoom	; Library y to new z and scale

	    x_est = (float(nfd_dc)-x_trans - cos(zrad)*x_dc + sin(zrad)*y_dc)*scale/512.0
	    y_est = (imaginary(nfd_dc)-y_trans-sin(zrad)*x_dc-cos(zrad)*y_dc)*scale/512.0
	    x_est = x_est*(principle_distance-z_est)/principle_distance
	    y_est = y_est*(principle_distance-z_est)/principle_distance

; Compensate x and y rotations for x and y translations

	phi_x = atan(y_est/(principle_distance-z_est))*180./!PI	; Get equivalent 
	phi_y = atan(x_est/(principle_distance-z_est))*180./!PI	; angles

	x_angle_corr = x_angle + cos(zrad)*phi_x - sin(zrad)*phi_y
	y_angle_corr = y_angle - sin(zrad)*phi_x - cos(zrad)*phi_y

	new_e(0) = x_est-tx_val*(txmax-txmin)-txmin
	new_e(1) = y_est-ty_val*(tymax-tymin)-tymin
	new_e(2) = z_est-tz_val*(tzmax-tzmin)-tzmin
	new_e(3) = x_angle_corr-rx_val*(rxmax-rxmin)-rxmin
	new_e(4) = y_angle_corr-ry_val*(rymax-rymin)-rymin
	new_e(5) = z_angle-rz_val*(rzmax-rzmin)-rzmin

	errors(its,*) = new_e
	estimates(its,*) = transpose([x_est,y_est,z_est,x_angle_corr,y_angle_corr,z_angle])
	weights(its) = dist1(order(its))

	new_cont = fft(shift(new_freq,-nsamp/2),1)
	contours(its,*) = new_cont

    endfor	; End of for its = 0,7 do ......

est:	siz = size(estimates)
	if (siz(0) eq 0 OR siz(1) eq 1) then begin
	    print,'No valid estimates of pose found!!!!'
	    return,1000
	endif

est1:	siz = size(estimates)
	cols = siz(1)
	if (cols eq 1) then begin
	    print,' Only one valid estimate.... '
	    weights = [1.0]
	    sum = 1.0
	    goto,out
	endif

; There are now up to its_max estimates of the object position/orientation. These need to be
; sorted so that those which result in poor contour fits are rejected or lightly weighted, and
; so that any estimates which are substantially different from the best estimate are rejected.
; The first step determines if there any outliers in the estimates, such that the euclidean
; distance between the ABD/ADD and INT/EXT estimates from the average is below some minimum.
; The second step takes the remaining estimates and forms a linear combination of them based
; on weightings inversely proportional to the fitted distance parameter "dist1" computed above.

	if (cols gt 1) then begin
	    ang_std = 10.0 ; Dummy "Spread" in angular estimates
	    i = -1
	    while (ang_std gt 1.0 AND i lt cols-2) do begin	  ; If spread is greater than "1 degree"
		i = i + 1					  ; eliminate poorer estimates
		x_ang_std = stdev(estimates(0:cols-1-i,3)-estimates(0,3)) ;STD of x angular estimates
		y_ang_std = stdev(estimates(0:cols-1-i,4)-estimates(0,4)) ;STD of y angular estimates
		ang_std = sqrt(x_ang_std^2+y_ang_std^2) 	  ; "Spread" in angular estimates
	    endwhile

	    if (i gt 0) then begin
		print,strcompress(string(i)+' estimates killed to decrease variance of angular estimate')
		print,''
		for j=0,i-1 do weights(cols-1-j) = 1000.0		; Tag for elimination
	    endif
	endif

	e_avg = fltarr(6)
	cluster = intarr(cols)
;	print,'weights=',weights
	sum = 1/(total(1/weights))	; Reciprocal of sum of reciprocals - "parallel combination"
	elim = 0
	for i=0,cols-1 do begin					; If a coefficient doesn't contrib-
	    if (sum/weights(i) gt 1./(2*cols)) then begin	; ute at least 1/(2*# of estimates)
		cluster(i) = 1					; it is ignored. This helps get rid
;		print,sum/weights(i),' valid weight',i		; of outliers.
	    endif else begin
		elim = 1
		cluster(i) = 0
;		print,sum/weights(i),' invalid weight',i
	    endelse
	endfor
;	print,where(cluster eq 1)

	estimates = estimates(where(cluster eq 1),*)
	errors = errors(where(cluster eq 1),*)
	weights = weights(where(cluster eq 1))

	if (elim) then begin				; If some weights are to be eliminated then
	    print,'Re-estimating weights .....'		; go back and recompute relative weightings
	    goto,est1
	endif

; Now form "optimal estimates"

out:	new_est = fltarr(1,6)
	new_cont = complexarr(nsamp)
	
	for i=0,cols-1 do begin
	    new_est = new_est + (sum/weights(i))*estimates(i,*)
	    new_cont = new_cont + transpose((sum/weights(i))*contours(i,*))
	endfor

	x_est = new_est(0) & y_est = new_est(1) & z_est = new_est(2)
	x_angle_corr = new_est(3) & y_angle_corr = new_est(4) & z_angle = new_est(5)

	zrad = z_angle*!PI/180.0
;	print,transpose(new_est)

	wset,4
	oplot,float(new_cont),imaginary(new_cont),color = 240,psym=1, $
		thick = 2.5,symsize=0.8

	diff = -nfd_angle*!PI/180.0
	new_cont = (new_cont)#complex(cos(diff),sin(diff))
	new_cont = (new_cont)*mag_match
	new_cont = new_cont+(dc_lib-complex(x_trans,y_trans))*complex(cos(zrad),sin(zrad))
	zoom1 = (principle_distance)/(principle_distance-z_est)
	new_cont = new_cont + $
		   complex(x_est*zoom1*512./scale+x_trans,y_est*zoom1*512./scale+y_trans)
	wset,0
	plots,float(new_cont),imaginary(new_cont)+y_pic_offset, $
	      color = 255,/device,thick=2.0			

	print,'Position/Orientation Errors .........'
	print,'x tran  =',x_est-tx_val*(txmax-txmin)-txmin
	print,'y tran  =',y_est-ty_val*(tymax-tymin)-tymin
	print,'z tran  =',z_est-tz_val*(tzmax-tzmin)-tzmin
	print,'x_angle =',x_angle_corr-rx_val*(rxmax-rxmin)-rxmin
	print,'y_angle =',y_angle_corr-ry_val*(rymax-rymin)-rymin
	print,'z_angle =',z_angle-rz_val*(rzmax-rzmin)-rzmin
	print,transpose(errors)
	print,''
;	print,transpose(errors_old)

	return, dist1(order(0))
	end