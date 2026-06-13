;============================================================================================
Compute moisture budget: use for pclimcontrol and piclim2xdust experiment
;============================================================================================
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
;============================================================================================
;============================================================================================
begin
;============================================================================================

in1   = addfile("piclim2xdust_hus.nc","r")
in2   = addfile("piclim2xdust_ua.nc","r")
in3   = addfile("piclim2xdust_va.nc","r")
in4   = addfile("pclim2xdust_omega.nc","r")


;============================================================================================
qq = short2flt(in1->hus(:,:,:,:))
uu =  short2flt(in2->ua(:,:,:,:))
vv =  short2flt(in3->va(:,:,:,:))
ww =  short2flt(in4->omega(:,:,:,:))


;qq&level = qq&level/100 
;;uu&level = uu&level/100 
;vv&level = vv&level/100 
;ww&level = ww&level/100 



  
   
q=qq(:,{1000:300},:,:)
u=uu(:,{1000:300},:,:)
v=vv(:,{1000:300},:,:)
w=ww(:,{1000:300},:,:)
;============================================================================================
lat =  q&lat
lon =  q&lon
time = q&time
plev = q&level
plev = plev*100       ;Change the pressure to Pa ?

nlat = dimsizes(lat)
nlon = dimsizes(lon)
ntim = dimsizes(time)
nlev = dimsizes(plev)
;============================================================================================
print("q:  min="+min(q)+"   max="+max(q))
print("u:  min="+min(u)+"   max="+max(u))
print("v:  min="+min(v)+"   max="+max(v))
print("w:  min="+min(w)+"   max="+max(w))

;============================================================================================
;==================================average===================================================
;============================================================================================
;============================================================================================
meanQ = dim_avg_n_Wrap(q,0)
meanU = dim_avg_n_Wrap(u,0)
meanV = dim_avg_n_Wrap(v,0)
meanW = dim_avg_n_Wrap(w,0)

meanQ_4d =conform(q,meanQ(:,:,:), (/1,2,3/) )
meanU_4d =conform(u,meanU(:,:,:), (/1,2,3/) )
meanV_4d =conform(v,meanV(:,:,:), (/1,2,3/) )
meanW_4d =conform(w,meanW(:,:,:), (/1,2,3/) )
;============================================================================================
;===================================anomaly==================================================
eddy_Q = q-meanQ_4d
eddy_U = u-meanU_4d
eddy_V = v-meanV_4d
eddy_W = w-meanW_4d

copy_VarCoords(q,eddy_Q)
copy_VarCoords(u,eddy_U)
copy_VarCoords(v,eddy_V)
copy_VarCoords(q,eddy_W)
;============================================================================================
;============================================================================================
;============================================================================================
   dlon = (lon(2)-lon(1))*0.0174533  
   dlat = (lat(2)-lat(1))*0.0174533 
;============================================================================================
;============================================================================================
dmeanQdX = new ( (/ntim,nlev,nlat,nlon/), "double", meanQ_4d@_FillValue)
deddyQdX = new ( (/ntim,nlev,nlat,nlon/), "double", eddy_Q@_FillValue)

dmeanQdY = new ( (/ntim,nlev,nlat,nlon/), "double", meanQ_4d@_FillValue)
deddyQdY = new ( (/ntim,nlev,nlat,nlon/), "double", eddy_Q@_FillValue)
;============================================================================================
;============================================================================================
do nl=0,nlat-1  
      dX = 6378388.0*cos(0.0174533*lat(nl))*dlon  ; constant at this lat

      dmeanQdX(:,:,nl:nl,:) = center_finite_diff_n (meanQ_4d(:,:,nl:nl,:), dX , True,0,3)
      deddyQdX(:,:,nl:nl,:) = center_finite_diff_n (eddy_Q(:,:,nl:nl,:), dX , True,0,3)

  end do

 copy_VarCoords(meanQ_4d, dmeanQdX)
 copy_VarCoords(eddy_Q, deddyQdX)
;============================================================================================
do nY=0,nlon-1 
 
   dY   = 6378388.0*dlat                    
      dmeanQdY(:,:,:,nY:nY) = center_finite_diff_n (meanQ_4d(:,:,:,nY:nY), dY , True,0,2)
      deddyQdY(:,:,:,nY:nY) = center_finite_diff_n (eddy_Q(:,:,:,nY:nY), dY , True,0,2)
end do

 copy_VarCoords(meanQ_4d, dmeanQdY)
 copy_VarCoords(eddy_Q, deddyQdY)


    printMinMax(dmeanQdX, False)
    printMinMax(dmeanQdY,False)

;============================================================================================
;=================================dq/dp==========dq'/dp======================================
;============================================================================================

dmeanQdP = center_finite_diff_n(meanQ_4d,plev,False,0,1)
deddyQdP = center_finite_diff_n(eddy_Q,plev,False,0,1)

;============================================================================================
;================================MBE terms vertical==========================================

meanW_deddyQdP = -1 *(meanW_4d*deddyQdP) 

eddyW_dmeanQdP = -1 *(eddy_W*dmeanQdP)

eddyW_deddyQdP = -1 *(eddy_W*deddyQdP)

copy_VarCoords(meanW_4d,meanW_deddyQdP)
copy_VarCoords(meanW_4d,eddyW_dmeanQdP)
copy_VarCoords(meanW_4d,eddyW_deddyQdP)

;============================================================================================
;================================MBE terms zonal=============================================

meanU_deddyQdX = -1 *(meanU_4d*deddyQdX) 

eddyU_dmeanQdX = -1 *(eddy_U*dmeanQdX)

eddyU_deddyQdX = -1 *(eddy_U*deddyQdX)

copy_VarCoords(meanW_4d,meanU_deddyQdX)
copy_VarCoords(meanW_4d,eddyU_dmeanQdX)
copy_VarCoords(meanW_4d,eddyU_deddyQdX)
;============================================================================================
;================================MBE terms Meridional========================================

meanV_deddyQdY = -1 *(meanV_4d*deddyQdY) 

eddyV_dmeanQdY = -1 *(eddy_V*dmeanQdY)

eddyV_deddyQdY = -1 *(eddy_V*deddyQdY)


copy_VarCoords(meanW_4d,meanV_deddyQdY)
copy_VarCoords(meanW_4d,eddyV_dmeanQdY)
copy_VarCoords(meanW_4d,eddyV_deddyQdY)

;============================================================================================
;================================MBE terms UV================================================
meanUV_deddyQdXdY = -1 *((meanU_4d*deddyQdX) + (meanV_4d*deddyQdY))

eddyUV_dmeanQdXdY = -1 *((eddy_U*dmeanQdX) + (eddy_V*dmeanQdY))

eddyUV_deddyQdXdY = -1 *((eddy_U*deddyQdX) + (eddy_V*deddyQdY))

copy_VarCoords(meanW_4d,meanUV_deddyQdXdY)
copy_VarCoords(meanW_4d,eddyUV_dmeanQdXdY)
copy_VarCoords(meanW_4d,eddyUV_deddyQdXdY)
;============================================================================================
;============================================================================================

	level0=q&level
	wantedlevel=ispan(1000, 300, 100)
	;wantedlevel=ispan(1000, 300, 100)							
	level_interval=wantedlevel													
	level_interval=10000														
	level_interval(0)=5000							         			 
	level_interval(dimsizes(level_interval)-1)=5000


;=================================================================================================================
;=================================================================================================================
;===================================================vertical integral==============================================
	W1=linint1_n_Wrap(level0, meanW_deddyQdP, False, wantedlevel, 0, 1)						
	meanW_deddyQdP_quint=dim_sum_n(W1*conform(W1, level_interval, 1)/9.8, 1) * 86400

	W2=linint1_n_Wrap(level0, eddyW_dmeanQdP, False, wantedlevel, 0, 1)						
	eddyW_dmeanQdP_quint=dim_sum_n(W2*conform(W2, level_interval, 1)/9.8, 1) * 86400


	W3=linint1_n_Wrap(level0, eddyW_deddyQdP, False, wantedlevel, 0, 1)						
	eddyW_deddyQdP_quint=dim_sum_n(W3*conform(W3, level_interval, 1)/9.8, 1) * 86400


meanW_deddyQdP_quint!0="time"
meanW_deddyQdP_quint!1="lat"
meanW_deddyQdP_quint!2="lon"
meanW_deddyQdP_quint&time=q&time
meanW_deddyQdP_quint&lat=q&lat
meanW_deddyQdP_quint&lon=q&lon


eddyW_dmeanQdP_quint!0="time"
eddyW_dmeanQdP_quint!1="lat"
eddyW_dmeanQdP_quint!2="lon"
eddyW_dmeanQdP_quint&time=q&time
eddyW_dmeanQdP_quint&lat=q&lat
eddyW_dmeanQdP_quint&lon=q&lon


eddyW_deddyQdP_quint!0="time"
eddyW_deddyQdP_quint!1="lat"
eddyW_deddyQdP_quint!2="lon"
eddyW_deddyQdP_quint&time=q&time
eddyW_deddyQdP_quint&lat=q&lat
eddyW_deddyQdP_quint&lon=q&lon




        ncdf = addfile("meanW_deddyQdP_quint.nc" ,"c")  
	ncdf->meanW_deddyQdP_quint = meanW_deddyQdP_quint 

        ncdf = addfile("eddyW_dmeanQdP_quint.nc" ,"c")  
	ncdf->eddyW_dmeanQdP_quint = eddyW_dmeanQdP_quint 

        ncdf = addfile("eddyW_deddyQdP_quint.nc" ,"c")  
	ncdf->eddyW_deddyQdP_quint = eddyW_deddyQdP_quint 

;=================================================================================================================
;=================================================================================================================
	U1=linint1_n_Wrap(level0, meanU_deddyQdX, False, wantedlevel, 0, 1)						
	meanU_deddyQdX_quint=dim_sum_n(U1*conform(U1, level_interval, 1)/9.8, 1) * 86400

	U2=linint1_n_Wrap(level0, eddyU_dmeanQdX, False, wantedlevel, 0, 1)						
	eddyU_dmeanQdX_quint=dim_sum_n(U2*conform(U2, level_interval, 1)/9.8, 1) * 86400 


	U3=linint1_n_Wrap(level0, eddyU_deddyQdX, False, wantedlevel, 0, 1)						
	eddyU_deddyQdX_quint=dim_sum_n(U3*conform(U3, level_interval, 1)/9.8, 1) * 86400 



meanU_deddyQdX_quint!0="time"
meanU_deddyQdX_quint!1="lat"
meanU_deddyQdX_quint!2="lon"
meanU_deddyQdX_quint&time=q&time
meanU_deddyQdX_quint&lat=q&lat
meanU_deddyQdX_quint&lon=q&lon


eddyU_dmeanQdX_quint!0="time"
eddyU_dmeanQdX_quint!1="lat"
eddyU_dmeanQdX_quint!2="lon"
eddyU_dmeanQdX_quint&time=q&time
eddyU_dmeanQdX_quint&lat=q&lat
eddyU_dmeanQdX_quint&lon=q&lon


eddyU_deddyQdX_quint!0="time"
eddyU_deddyQdX_quint!1="lat"
eddyU_deddyQdX_quint!2="lon"
eddyU_deddyQdX_quint&time=q&time
eddyU_deddyQdX_quint&lat=q&lat
eddyU_deddyQdX_quint&lon=q&lon



     ;   ncdf = addfile("meanU_deddyQdX_quint.nc" ,"c")  
	;ncdf->meanU_deddyQdX_quint = meanU_deddyQdX_quint 

      ;  ncdf = addfile("eddyU_dmeanQdX_quint.nc" ,"c")  
	;ncdf->eddyU_dmeanQdX_quint = eddyU_dmeanQdX_quint 

      ;  ncdf = addfile("eddyU_deddyQdX_quint.nc" ,"c")  
	;ncdf->eddyU_deddyQdX_quint = eddyU_deddyQdX_quint 

;=================================================================================================================
;=================================================================================================================
	V1=linint1_n_Wrap(level0, meanV_deddyQdY, False, wantedlevel, 0, 1)						
	meanV_deddyQdY_quint=dim_sum_n(V1*conform(V1, level_interval, 1)/9.8, 1) * 86400

	V2=linint1_n_Wrap(level0, eddyV_dmeanQdY, False, wantedlevel, 0, 1)						
	eddyV_dmeanQdY_quint=dim_sum_n(V2*conform(V2, level_interval, 1)/9.8, 1) * 86400


	V3=linint1_n_Wrap(level0, eddyV_deddyQdY, False, wantedlevel, 0, 1)						
	eddyV_deddyQdY_quint=dim_sum_n(V3*conform(V3, level_interval, 1)/9.8, 1) * 86400


meanV_deddyQdY_quint!0="time"
meanV_deddyQdY_quint!1="lat"
meanV_deddyQdY_quint!2="lon"
meanV_deddyQdY_quint&time=q&time
meanV_deddyQdY_quint&lat=q&lat
meanV_deddyQdY_quint&lon=q&lon


eddyV_dmeanQdY_quint!0="time"
eddyV_dmeanQdY_quint!1="lat"
eddyV_dmeanQdY_quint!2="lon"
eddyV_dmeanQdY_quint&time=q&time
eddyV_dmeanQdY_quint&lat=q&lat
eddyV_dmeanQdY_quint&lon=q&lon


eddyV_deddyQdY_quint!0="time"
eddyV_deddyQdY_quint!1="lat"
eddyV_deddyQdY_quint!2="lon"
eddyV_deddyQdY_quint&time=q&time
eddyV_deddyQdY_quint&lat=q&lat
eddyV_deddyQdY_quint&lon=q&lon




      ;  ncdf = addfile("meanV_deddyQdY_quint.nc" ,"c")  
	;ncdf->meanV_deddyQdY_quint = meanV_deddyQdY_quint 

       ; ncdf = addfile("eddyV_dmeanQdY_quint.nc" ,"c")  
	;ncdf->eddyV_dmeanQdY_quint = eddyV_dmeanQdY_quint 

      ;  ncdf = addfile("eddyV_deddyQdY_quint.nc" ,"c")  
	;ncdf->eddyV_deddyQdY_quint = eddyV_deddyQdY_quint 

;=================================================================================================================
;=================================================================================================================
	UV1=linint1_n_Wrap(level0, meanUV_deddyQdXdY, False, wantedlevel, 0, 1)						
	meanUV_deddyQdXdY_quint=dim_sum_n(UV1*conform(UV1, level_interval, 1)/9.8, 1)  * 86400

	UV2=linint1_n_Wrap(level0, eddyUV_dmeanQdXdY, False, wantedlevel, 0, 1)						
	eddyUV_dmeanQdXdY_quint=dim_sum_n(UV2*conform(UV2, level_interval, 1)/9.8, 1) * 86400


	UV3=linint1_n_Wrap(level0, eddyUV_deddyQdXdY, False, wantedlevel, 0, 1)						
	eddyUV_deddyQdXdY_quint=dim_sum_n(UV3*conform(UV3, level_interval, 1)/9.8, 1) * 86400


meanUV_deddyQdXdY_quint!0="time"
meanUV_deddyQdXdY_quint!1="lat"
meanUV_deddyQdXdY_quint!2="lon"
meanUV_deddyQdXdY_quint&time=q&time
meanUV_deddyQdXdY_quint&lat=q&lat
meanUV_deddyQdXdY_quint&lon=q&lon


eddyUV_dmeanQdXdY_quint!0="time"
eddyUV_dmeanQdXdY_quint!1="lat"
eddyUV_dmeanQdXdY_quint!2="lon"
eddyUV_dmeanQdXdY_quint&time=q&time
eddyUV_dmeanQdXdY_quint&lat=q&lat
eddyUV_dmeanQdXdY_quint&lon=q&lon


eddyUV_deddyQdXdY_quint!0="time"
eddyUV_deddyQdXdY_quint!1="lat"
eddyUV_deddyQdXdY_quint!2="lon"
eddyUV_deddyQdXdY_quint&time=q&time
eddyUV_deddyQdXdY_quint&lat=q&lat
eddyUV_deddyQdXdY_quint&lon=q&lon


;        ncdf = addfile("meanUV_deddyQdXdY_quint.nc" ,"c")  
;	ncdf->meanUV_deddyQdXdY_quint = meanUV_deddyQdXdY_quint 

 ;       ncdf = addfile("eddyUV_dmeanQdXdY_quint.nc" ,"c")  
;	ncdf->eddyUV_dmeanQdXdY_quint = eddyUV_dmeanQdXdY_quint 
 ;       ncdf = addfile("eddyUV_deddyQdXdY_quint.nc" ,"c")  
;	ncdf->eddyUV_deddyQdXdY_quint = eddyUV_deddyQdXdY_quint 


;=================================================================================================================
;=================================================================================================================
;=================================================================================================================
;=================================================================================================================
end

