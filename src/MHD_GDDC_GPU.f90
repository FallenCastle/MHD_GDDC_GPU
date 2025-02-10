module setup
    implicit none

    integer :: npx,npy,npz
    integer :: iter,maxiter
    integer :: nVar = 8, bound_type(4)
    integer, allocatable::neighbxp(:,:),neighbxm(:,:),&
                          neighbyp(:,:),neighbym(:,:),&
                          neighbx(:,:),neighby(:,:)
    double precision :: hx,hy,dt,ctime,tf,residglob,eta,xmin,xmax,ymin,ymax
    double precision,allocatable :: ub(:,:,:),flux1(:,:,:),&
            flux2(:,:,:),resid(:,:,:)
    double precision,allocatable :: dB(:,:),Flagp(:,:,:),Flagm(:,:,:)
    double precision:: gam
    double precision :: PI = 4.d0*DATAN(1.d0)
    
    double precision::timeStart, timeStart1,timeStop1, timeStart2,timeStop2,timeEnd
    double precision,dimension(8) :: uInlet
    character*64 :: namerestart
    integer :: restart
    character(len=100):: folder_path
    character(len=10):: boundary(4)
    integer :: date_values(8), ierr, NTF

end module

module setup_device
    implicit none

    integer, constant:: npx_d, npy_d
    integer, constant:: nVar_d
    integer,device::bound_type_d(4)
    
    double precision, constant:: gam_d,x0_d,y0_d,radius_d,dt_d,eta_min_d,hx_d,hy_d,eta_d
    double precision, device:: residglob_d
    double precision, device, allocatable:: ub_d(:,:,:), resid_d(:,:,:),&
            ub0_d(:,:,:), uba_d(:,:,:), dB_d(:,:),uInlet_d(:)
    integer,device,allocatable::neighbxp_d(:,:),neighbxm_d(:,:),&
                                neighbyp_d(:,:),neighbym_d(:,:),&
                                neighbx_d(:,:),neighby_d(:,:)
    
end module setup_device

module kernel_function
    contains
    
    subroutine SSPRK_time_stepping_gpu
        use setup
        use setup_device
        use cudafor
        use ieee_arithmetic
        implicit none
    
        integer:: tilex=32, tiley=8,&
                  bLkx,bLky,gridSolid
        integer:: istat,ipx,ipy
        integer(c_int)::ierr_
    
    
        type(dim3)::grid0,gridx,gridy,block
        type(cudaEvent):: startEvent, stopEvent
    
        block = dim3(tilex,tiley,1)
        
        bLkx = (npx+tilex-1)/tilex
        bLky = (npy+tiley-1)/tiley
        grid0 = dim3(bLkx,bLky,1)
    
        bLkx = (npx+1+tilex-1)/tilex
        bLky = (npy  +tiley-1)/tiley
        gridx = dim3(bLkx,bLky,1)
    
        bLkx = (npx  +tilex-1)/tilex
        bLky = (npy+1+tiley-1)/tiley
        gridy = dim3(bLkx,bLky,1)
    
        istat = cudaEventCreate(startEvent)
        istat = cudaEventCreate(stopEvent)
        !!!=========================================
    
        ub0_d = ub_d
        
        resid_d = 0.d0
        
        istat = cudaEventRecord(startEvent,0)
        call compute_fluxF<<<gridx,block>>>()
        call compute_fluxG<<<gridy,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        
        istat = cudaEventRecord(startEvent,0)
        call calDivB_gpu<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
    
        istat = cudaEventRecord(startEvent,0)
        call ComputeSources_GD_gpu<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        
        istat = cudaEventRecord(startEvent,0)
        call ssprk_stage_1<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        !!--------------------------------------------
        resid_d = 0.d0
    
        istat = cudaEventRecord(startEvent,0)
        call compute_fluxF<<<gridx,block>>>()
        call compute_fluxG<<<gridy,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
    
        istat = cudaEventRecord(startEvent,0)
        call calDivB_gpu<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        
        istat = cudaEventRecord(startEvent,0)
        call ComputeSources_GD_gpu<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        
        istat = cudaEventRecord(startEvent,0)
        call ssprk_stage_2<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        !!--------------------------------------------
    
        uba_d = ub_d
        resid_d = 0.d0
    
        istat = cudaEventRecord(startEvent,0)
        call compute_fluxF<<<gridx,block>>>()
        call compute_fluxG<<<gridy,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
    
        istat = cudaEventRecord(startEvent,0)
        call calDivB_gpu<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        
        istat = cudaEventRecord(startEvent,0)
        call ComputeSources_GD_gpu<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        
        istat = cudaEventRecord(startEvent,0)
        call ssprk_stage_3<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
    
        !!--------------------------------------------
    
        resid_d = 0.d0
    
        istat = cudaEventRecord(startEvent,0)
        call compute_fluxF<<<gridx,block>>>()
        call compute_fluxG<<<gridy,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
    
        istat = cudaEventRecord(startEvent,0)
        call calDivB_gpu<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        
        istat = cudaEventRecord(startEvent,0)
        call ComputeSources_GD_gpu<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        
        istat = cudaEventRecord(startEvent,0)
        call ssprk_stage_4<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
    
        !!--------------------------------------------
    
        resid_d = 0.d0
    
        istat = cudaEventRecord(startEvent,0)
        call compute_fluxF<<<gridx,block>>>()
        call compute_fluxG<<<gridy,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
    
        istat = cudaEventRecord(startEvent,0)
        call calDivB_gpu<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        
        istat = cudaEventRecord(startEvent,0)
        call ComputeSources_GD_gpu<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        
        istat = cudaEventRecord(startEvent,0)
        call ssprk_stage_5<<<grid0,block>>>()
        istat = cudaEventRecord(stopEvent,0)
        istat = cudaEventSynchronize(stopEvent)
        !!--------------------------------------------
        
        ctime = ctime + dt
    
        
    end subroutine SSPRK_time_stepping_gpu
    
    attributes(global) subroutine compute_fluxF()
        use setup_device
        implicit none
    
        integer::i,j,i1,im3,im2,im1,ip1,ip2,index,pt_index
        double precision::uim3(8),uim2(8),uim1(8),ui(8),uip1(8),uip2(8)
        double precision::uavg(8), Rmat(8,8),&
                          Rinv(8,8), diag(8,8),&
                          fp_stencil(8,5), fm_stencil(8,5),&
                          fluxp(8), fluxm(8), num_flux(8),res,nf(3),&
                          ub_stencil(8,5)
    
        i=threadidx%x+(blockidx%x-1)*blockdim%x
        j=threadidx%y+(blockidx%y-1)*blockdim%y
    
        if(i<=npx_d+1 .and. j<=npy_d)then
            
            nf(1) = 1.d0
            nf(2) = 0.d0
            nf(3) = 0.d0
    
            if(i==1)then
                if(bound_type_d(1)==1)then
                    uavg(:) = uInlet_d(:)
                elseif(bound_type_d(1)==2)then
                    uavg(:) = ub_d(1,j,:)
                elseif(bound_type_d(1)==3)then
                    uavg(:) = 0.5d0*(ub_d(1,j,:)+ub_d(npx_d,j,:))
                elseif(bound_type_d(1)==4)then
                    uavg(:) = ub_d(1,j,:)
                endif
            elseif(i==npx_d+1)then
                if(bound_type_d(2)==1)then
                    uavg(:) = uInlet_d(:)
                elseif(bound_type_d(2)==2)then
                    uavg(:) = ub_d(npx_d,j,:)
                elseif(bound_type_d(2)==3)then
                    uavg(:) = 0.5d0*(ub_d(1,j,:)+ub_d(npx_d,j,:))
                elseif(bound_type_d(2)==4)then
                    uavg(:) = ub_d(npx_d,j,:)
                endif
            else
                uavg(:) = 0.5d0*(ub_d(i,j,:) + ub_d(i-1,j,:))
            endif
    
            call compute_evr_gpu(uavg,nf(1),nf(2),nf(3),Rmat)
            call compute_evl_gpu(uavg,nf(1),nf(2),nf(3),Rinv)
            
            do index = 1,5
                if(neighbxp_d(i,index)==0)then
                    ub_stencil(:,index) = uInlet_d(:)
                else
                    ub_stencil(:,index) = ub_d(neighbxp_d(i,index),j,:)
                endif
            enddo
            call flux_split_gpu(ub_stencil(:,1),nf,fp_stencil(:,1),1)
            call flux_split_gpu(ub_stencil(:,2),nf,fp_stencil(:,2),1)
            call flux_split_gpu(ub_stencil(:,3),nf,fp_stencil(:,3),1)
            call flux_split_gpu(ub_stencil(:,4),nf,fp_stencil(:,4),1)
            call flux_split_gpu(ub_stencil(:,5),nf,fp_stencil(:,5),1)

            call reconstruct_gpu(fp_stencil,Rmat,Rinv,fluxp)

            do index = 1,5
                if(neighbxm_d(i,index)==0)then
                    ub_stencil(:,index) = uInlet_d(:)
                else
                    ub_stencil(:,index) = ub_d(neighbxm_d(i,index),j,:)
                endif
            enddo
            call flux_split_gpu(ub_stencil(:,1),nf,fm_stencil(:,1),-1)
            call flux_split_gpu(ub_stencil(:,2),nf,fm_stencil(:,2),-1)
            call flux_split_gpu(ub_stencil(:,3),nf,fm_stencil(:,3),-1)
            call flux_split_gpu(ub_stencil(:,4),nf,fm_stencil(:,4),-1)
            call flux_split_gpu(ub_stencil(:,5),nf,fm_stencil(:,5),-1)
            
            call reconstruct_gpu(fm_stencil,Rmat,Rinv,fluxm)
            
            !!----------------------------------------------------
            num_flux(:) = fluxp(:)+fluxm(:)

            res=atomicadd(resid_d(i-1,j,1), -num_flux(1)/hx_d)
            res=atomicadd(resid_d(i,  j,1),  num_flux(1)/hx_d)

            res=atomicadd(resid_d(i-1,j,2), -num_flux(2)/hx_d)
            res=atomicadd(resid_d(i,  j,2),  num_flux(2)/hx_d)

            res=atomicadd(resid_d(i-1,j,3), -num_flux(3)/hx_d)
            res=atomicadd(resid_d(i,  j,3),  num_flux(3)/hx_d)

            res=atomicadd(resid_d(i-1,j,4), -num_flux(4)/hx_d)
            res=atomicadd(resid_d(i,  j,4),  num_flux(4)/hx_d)

            res=atomicadd(resid_d(i-1,j,5), -num_flux(5)/hx_d)
            res=atomicadd(resid_d(i,  j,5),  num_flux(5)/hx_d)

            res=atomicadd(resid_d(i-1,j,6), -num_flux(6)/hx_d)
            res=atomicadd(resid_d(i,  j,6),  num_flux(6)/hx_d)

            res=atomicadd(resid_d(i-1,j,7), -num_flux(7)/hx_d)
            res=atomicadd(resid_d(i,  j,7),  num_flux(7)/hx_d)
 
            res=atomicadd(resid_d(i-1,j,8), -num_flux(8)/hx_d)
            res=atomicadd(resid_d(i,  j,8),  num_flux(8)/hx_d)
    
        endif
    
    end subroutine compute_fluxF
    
    attributes(global) subroutine compute_fluxG()
        use setup_device
        implicit none
    
        integer::i,j,j1,jm3,jm2,jm1,jp1,jp2,index,pt_index
        double precision::ujm3(8),ujm2(8),ujm1(8),uj(8),ujp1(8),ujp2(8)
        double precision::uavg(8), Rmat(8,8),&
                          Rinv(8,8), diag(8,8),&
                          fp_stencil(8,5), fm_stencil(8,5),&
                          fluxp(8), fluxm(8), num_flux(8),res,nf(3),&
                          ub_stencil(8,5)
    
        i=threadidx%x+(blockidx%x-1)*blockdim%x
        j=threadidx%y+(blockidx%y-1)*blockdim%y
        if(i<=npx_d .and. j<=npy_d+1)then
            nf(1) = 0.d0
            nf(2) = 1.d0
            nf(3) = 0.d0
    
            if(j==1)then
                if(bound_type_d(3)==1)then
                    uavg(:)=uInlet_d(:)
                elseif(bound_type_d(3)==2)then
                    uavg(:)=ub_d(i,1,:)
                elseif(bound_type_d(3)==3)then
                    uavg(:)=0.5d0*(ub_d(i,1,:)+ub_d(i,npy_d,:))
                elseif(bound_type_d(3)==4)then
                    uavg(:)=ub_d(i,1,:)
                endif
            elseif(j==npy_d+1)then
                if(bound_type_d(4)==1)then
                    uavg(:)=uInlet_d(:)
                elseif(bound_type_d(4)==2)then
                    uavg(:)=ub_d(i,npy_d,:)
                elseif(bound_type_d(4)==3)then
                    uavg(:)=0.5d0*(ub_d(i,1,:)+ub_d(i,npy_d,:))
                elseif(bound_type_d(4)==4)then
                    uavg(:)=ub_d(i,npy_d,:)
                endif
            else
                uavg(:) = 0.5d0*(ub_d(i,j,:) + ub_d(i,j-1,:))
            endif
        
            call compute_evr_gpu(uavg,nf(1),nf(2),nf(3),Rmat)
            call compute_evl_gpu(uavg,nf(1),nf(2),nf(3),Rinv)

            do index = 1,5
                if(neighbyp_d(j,index)==0)then
                    ub_stencil(:,index) = uInlet_d(:)
                else
                    ub_stencil(:,index) = ub_d(i,neighbyp_d(j,index),:)
                endif
            enddo
            call flux_split_gpu(ub_stencil(:,1),nf,fp_stencil(:,1),1)
            call flux_split_gpu(ub_stencil(:,2),nf,fp_stencil(:,2),1)
            call flux_split_gpu(ub_stencil(:,3),nf,fp_stencil(:,3),1)
            call flux_split_gpu(ub_stencil(:,4),nf,fp_stencil(:,4),1)
            call flux_split_gpu(ub_stencil(:,5),nf,fp_stencil(:,5),1)

            call reconstruct_gpu(fp_stencil,Rmat,Rinv,fluxp)

            do index = 1,5
                if(neighbym_d(j,index)==0)then
                    ub_stencil(:,index) = uInlet_d(:)
                else
                    ub_stencil(:,index) = ub_d(i,neighbym_d(j,index),:)
                endif
            enddo
            call flux_split_gpu(ub_stencil(:,1),nf,fm_stencil(:,1),-1)
            call flux_split_gpu(ub_stencil(:,2),nf,fm_stencil(:,2),-1)
            call flux_split_gpu(ub_stencil(:,3),nf,fm_stencil(:,3),-1)
            call flux_split_gpu(ub_stencil(:,4),nf,fm_stencil(:,4),-1)
            call flux_split_gpu(ub_stencil(:,5),nf,fm_stencil(:,5),-1)
            
            call reconstruct_gpu(fm_stencil,Rmat,Rinv,fluxm)
            !!------------------------------------------------------
            
            num_flux(:) = fluxp(:)+fluxm(:)
    
            res=atomicadd(resid_d(i,j-1,1), -num_flux(1)/hy_d)
            res=atomicadd(resid_d(i,  j,1),  num_flux(1)/hy_d)
    
            res=atomicadd(resid_d(i,j-1,2), -num_flux(2)/hy_d)
            res=atomicadd(resid_d(i,  j,2),  num_flux(2)/hy_d)
    
            res=atomicadd(resid_d(i,j-1,3), -num_flux(3)/hy_d)
            res=atomicadd(resid_d(i,  j,3),  num_flux(3)/hy_d)
    
            res=atomicadd(resid_d(i,j-1,4), -num_flux(4)/hy_d)
            res=atomicadd(resid_d(i,  j,4),  num_flux(4)/hy_d)
    
            res=atomicadd(resid_d(i,j-1,5), -num_flux(5)/hy_d)
            res=atomicadd(resid_d(i,  j,5),  num_flux(5)/hy_d)
    
            res=atomicadd(resid_d(i,j-1,6), -num_flux(6)/hy_d)
            res=atomicadd(resid_d(i,  j,6),  num_flux(6)/hy_d)
    
            res=atomicadd(resid_d(i,j-1,7), -num_flux(7)/hy_d)
            res=atomicadd(resid_d(i,  j,7),  num_flux(7)/hy_d)
    
            res=atomicadd(resid_d(i,j-1,8), -num_flux(8)/hy_d)
            res=atomicadd(resid_d(i,  j,8),  num_flux(8)/hy_d)
    
        endif
    
    end subroutine compute_fluxG
    
    attributes(device) subroutine flux_split_gpu(var,nf,flux,wind)
        use setup_device,only:gam_d,nVar_d
        implicit none
    
        double precision :: var(8),flux(8),nf(3)
        integer :: wind
        double precision :: cf,rho,rhou,u,rhov,v,rhow,w,pr,Bx,By,Bz,unf,magnorm,NORM_B,bnf
    
        magnorm = dsqrt(nf(1)**2+nf(2)**2+nf(3)**2)
        nf(1) = nf(1)/magnorm
        nf(2) = nf(2)/magnorm
        nf(3) = nf(3)/magnorm
    
        rho = var(1)
        rhou= var(2)
        rhov= var(3)
        rhow= var(4)
        u   = rhou/rho
        v   = rhov/rho
        w   = rhow/rho
        Bx  = var(6)
        By  = var(7)
        Bz  = var(8)
        NORM_B = Bx**2+By**2+Bz**2
    
        pr  = (gam_d-1.d0)*(var(5) &
                -0.5d0*(rhou**2+rhov**2+rhow**2)/rho-0.5d0*NORM_B)
    
    
        unf = u*nf(1)+v*nf(2)+w*nf(3)
        bnf = Bx*nf(1)+By*nf(2)+Bz*nf(3)
    
        flux(1) = rho*unf
        flux(2) = rho*u*unf + (pr+0.5d0*NORM_B)*nf(1) - Bx*bnf
        flux(3) = rho*v*unf + (pr+0.5d0*NORM_B)*nf(2) - By*bnf
        flux(4) = rho*w*unf + (pr+0.5d0*NORM_B)*nf(3) - Bz*bnf
        flux(5) = unf*(var(5)+pr+0.5*NORM_B) - (u*Bx+v*By+w*Bz)*bnf
        flux(6) = Bx*unf - u*bnf
        flux(7) = By*unf - v*bnf
        flux(8) = Bz*unf - w*bnf
    
        cf = dsqrt(0.5*(gam_d*pr/rho + NORM_B/rho &
             + dsqrt((gam_d*pr/rho + NORM_B/rho)**2.d0 &
             - 4.d0*gam_d*pr*bnf**2.d0/(rho**2.d0))))
    
        if (wind==1) then
            flux = 0.5*(flux + (dabs(unf)+cf)*var)
        else if (wind==-1) then
            flux = 0.5*(flux - (dabs(unf)+cf)*var)
        end if
    
    end subroutine flux_split_gpu
    
    attributes(device) subroutine compute_evr_gpu(Qi,n1,n2,n3,evr)
            use setup_device, only : gam_d
            double precision,dimension(8),intent(in) :: Qi
            double precision :: rho,u1,u2,u3,p,B1,B2,B3
            double precision,intent(in) :: n1,n2,n3
            double precision :: t1,t2,t3,tnorm
            double precision,dimension(8,8),intent(out) :: evr
            double precision,dimension(8,8) :: ru
            integer :: m
            double precision :: rhosq,a2,a,d,cf,cs,beta1,g1,bN,sqg2,sq12
            double precision :: alphas,alphaf,sqpr,sqpor,sq1og,bst,TxN1,TxN2,TxN3
            double precision :: b1s,b2s,b3s
    
            rho = Qi(1)
            u1  = Qi(2)/Qi(1)
            u2  = Qi(3)/Qi(1)
            u3  = Qi(4)/Qi(1)
            B1  = Qi(6)
            B2  = Qi(7)
            B3  = Qi(8)
            p = (gam_d-1.d0)*(Qi(5)-0.5d0*(Qi(2)**2+Qi(3)**2+Qi(4)**2)/Qi(1)&
                    -0.5d0*(B1**2+B2**2+B3**2))
            !
            t1 = n2*B3-n3*B2
            t2 = n3*B1-n1*B3
            t3 = n1*B2-n2*B1
            tnorm = dsqrt(t1**2+t2**2+t3**2)
            t1 = t1/tnorm
            t2 = t2/tnorm
            t3 = t3/tnorm
            !
            !
            rhosq = dsqrt(rho)
            g1=gam_d-1.d0
            a2 = dabs(gam_d*p/rho)
            a = dsqrt(a2)
    
            sqg2=dsqrt(1.d0/(2.d0*gam_d))
    
            sq12 = dsqrt(0.5d0)
    
            sqpr = dsqrt(abs(p))/rho
    
            sqpor = dsqrt(abs(p/rho))
    
            sq1og = dsqrt(1.d0/gam_d)
    
            b1s=B1/rhosq
            b2s=B2/rhosq
            b3s=B3/rhosq
    
            bN = b1s*n1+b2s*n2+b3s*n3
    
            d = a2 + (b1s**2 + b2s**2 + b3s**2)
            cf = dsqrt(0.5d0*dabs(d+dsqrt(d**2-4.d0*a2*(bN)**2)))
            cs = dsqrt(0.5d0*dabs(d-dsqrt(d**2-4.d0*a2*(bN)**2)))
    
            beta1 = dsign(1.d0,(b1s*n1+b2s*n2+b3s*n3)*1.d0)
    
            if ( dabs(cf*cf-cs*cs).le. 1.d-12) then
                alphaf = dsin(datan(1.d0))
                alphas = dcos(datan(1.d0))
            else
                alphaf = sqrt(abs(a2 - cs*cs))/sqrt(abs(cf*cf-cs*cs))
                alphas = sqrt(abs(cf*cf - a2))/sqrt(abs(cf*cf-cs*cs))
            endif
    
            TxN1=n3*t2-n2*t3
            TxN2=n1*t3-n3*t1
            TxN3=n2*t1-n1*t2
            ru(1,1)   =sqrt(g1/gam_d)*rhosq
            ru(2,1)   =0.d0
            ru(3,1)   =0.d0
            ru(4,1)   =0.d0
            ru(5,1)   =0.d0
            ru(6,1)   =0.d0
            ru(7,1)   =0.d0
            ru(8,1)   =0.d0

            ru(1,2)   =0.d0
            ru(2,2)   =0.d0
            ru(3,2)   =0.d0
            ru(4,2)   =0.d0
            ru(5,2)   =0.d0
            ru(6,2)   =sq1og*a*n1
            ru(7,2)   =sq1og*a*n2
            ru(8,2)   =sq1og*a*n3
            
            ru(1,3)   =0.d0
            ru(2,3)   =-sq12*(sqpr*(TxN1))
            ru(3,3)   =-sq12*(sqpr*(TxN2))
            ru(4,3)   =-sq12*(sqpr*(TxN3))
            ru(5,3)   =0.d0
            ru(6,3)   =sq12*sqpor*(TxN1)
            ru(7,3)   =sq12*sqpor*(TxN2)
            ru(8,3)   =sq12*sqpor*(TxN3)
    
            ru(1,4)   =ru(1,3)
            ru(2,4)   =-ru(2,3)
            ru(3,4)   =-ru(3,3)
            ru(4,4)   =-ru(4,3)
            ru(5,4)   =ru(5,3)
            ru(6,4)   =ru(6,3)
            ru(7,4)   =ru(7,3)
            ru(8,4)   =ru(8,3)
    
            bst = (b1s*t1+b2s*t2+b3s*t3)
    
            ru(1,5)   =sqg2*alphaf*rhosq
            ru(2,5)   =sqg2*((alphaf*a2*n1+alphas*a*((bst)*n1-(bN)*t1)))/(rhosq*cf)
            ru(3,5)   =sqg2*((alphaf*a2*n2+alphas*a*((bst)*n2-(bN)*t2)))/(rhosq*cf)
            ru(4,5)   =sqg2*((alphaf*a2*n3+alphas*a*((bst)*n3-(bN)*t3)))/(rhosq*cf)
            ru(5,5)   =sqg2*alphaf*rhosq*a2
            ru(6,5)   =sqg2*alphas*a*t1
            ru(7,5)   =sqg2*alphas*a*t2
            ru(8,5)   =sqg2*alphas*a*t3
    
            ru(1,6) = ru(1,5)
            ru(2,6) = -ru(2,5)
            ru(3,6) = -ru(3,5)
            ru(4,6) = -ru(4,5)
            ru(5,6) = ru(5,5)
            ru(6,6) = ru(6,5)
            ru(7,6) = ru(7,5)
            ru(8,6) = ru(8,5)
    
            ru(1,7) =sqg2*alphas*rhosq
            ru(2,7) =beta1*sqg2*(alphaf*cf**2*t1+a*n1*alphas*(bN))/(rhosq*cf)
            ru(3,7) =beta1*sqg2*(alphaf*cf**2*t2+alphas*a*(bN)*n2)/(rhosq*cf)
            ru(4,7) =beta1*sqg2*(alphaf*cf**2*t3+alphas*a*(bN)*n3)/(rhosq*cf)
            ru(5,7) =a**2*sqg2*alphas*rhosq
            ru(6,7) =-sqg2*alphaf*a*t1
            ru(7,7) =-sqg2*alphaf*a*t2
            ru(8,7) =-sqg2*alphaf*a*t3

            ru(1,8) =ru(1,7)
            ru(2,8) =-ru(2,7)
            ru(3,8) =-ru(3,7)
            ru(4,8) =-ru(4,7)
            ru(5,8) =ru(5,7)
            ru(6,8) =ru(6,7)
            ru(7,8) =ru(7,7)
            ru(8,8) =ru(8,7)

            do m=1,8
    
                evr(1,m)=ru(1,m)/g1
                evr(2,m)=(ru(1,m)*u1 + ru(2,m)*rho)/g1
                evr(3,m)=(ru(1,m)*u2 + ru(3,m)*rho)/g1
                evr(4,m)=(ru(1,m)*u3 + ru(4,m)*rho)/g1
                evr(5,m)=(ru(5,m)/g1+B1*ru(6,m)+B2*ru(7,m)+B3*ru(8,m)+0.5d0*ru(1,m) &
                        *(u1**2+u2**2+u3**2) + ru(2,m)*u1*rho + ru(3,m)*u2*rho     &
                        +ru(4,m)*u3*rho)/g1
                evr(6,m)=ru(6,m)/g1
                evr(7,m)=ru(7,m)/g1
                evr(8,m)=ru(8,m)/g1
    
            enddo
    
        end subroutine compute_evr_gpu
    
        attributes(device) subroutine compute_evl_gpu(Qi,n1,n2,n3,evl)
            use setup_device,only : gam_d
            double precision,dimension(8),intent(in) :: Qi
            double precision :: rho,u1,u2,u3,p,B1,B2,B3,n1,n2,n3
            double precision :: t1,t2,t3,tnorm
            double precision,dimension(8,8),intent(out) :: evl
            double precision,dimension(8,8) :: lu
            integer :: m
            double precision :: rhosq,twosq,a,a2,d,cf,cs,beta1
            double precision :: alphaf,alphas,um2,nen,nen2,nen31,BN,nen3
            double precision :: nen51,nen52,nen71,nen72
            double precision :: g1,psq,sqgam,gu1sq
            double precision :: b1s,b2s,b3s,TxN1,TxN2,TxN3,BT,BNs
            double precision :: Term51,Term52,Term53,Term54,Term55,Term56,Term57
            double precision :: Term71,Term72,Term73,Term74,Term75,Term76,Term77
    
            rho = Qi(1)
            u1  = Qi(2)/Qi(1)
            u2  = Qi(3)/Qi(1)
            u3  = Qi(4)/Qi(1)
            B1  = Qi(6)
            B2  = Qi(7)
            B3  = Qi(8)
            p = (gam_d-1.d0)*(Qi(5)-0.5d0*(Qi(2)**2+Qi(3)**2+Qi(4)**2)/Qi(1)&
                    -0.5d0*(B1**2+B2**2+B3**2))
            
            t1 = n2*B3-n3*B2
            t2 = n3*B1-n1*B3
            t3 = n1*B2-n2*B1
            tnorm = dsqrt(t1**2+t2**2+t3**2)
            t1 = t1/tnorm
            t2 = t2/tnorm
            t3 = t3/tnorm
            
            rhosq = dsqrt(rho)
            g1=gam_d-1.d0
            a2 = dabs(gam_d*p/rho)
            a = dsqrt(a2)
    
            b1s=B1/rhosq
            b2s=B2/rhosq
            b3s=B3/rhosq
    
            BNs = b1s*n1+b2s*n2+b3s*n3
    
            BN = (B1*n1+B2*n2+B3*n3)
    
            d = a2 + (b1s**2 + b2s**2 + b3s**2)
            cf = dsqrt(0.5d0*dabs(d+dsqrt(d**2-4.d0*a2*(BNs)**2)))
            cs = dsqrt(0.5d0*dabs(d-dsqrt(d**2-4.d0*a2*(BNs)**2)))
    
            if ( dabs(cf*cf-cs*cs).le. 1.d-12) then
                alphaf = dsin(datan(1.d0))
                alphas = dcos(datan(1.d0))
            else
                alphaf = sqrt(abs(a2 - cs*cs))/sqrt(abs(cf*cf-cs*cs))
                alphas = sqrt(abs(cf*cf - a2))/sqrt(abs(cf*cf-cs*cs))
            endif
    
            psq   = dsqrt(abs(p))
            twosq = dsqrt(2.d0)
            um2 = 0.5d0*(u1**2 + u2**2 + u3**2)
    
            beta1 = dsign(1.d0,(BNs)*1.d0)
    
            sqgam = dsqrt(g1/gam_d)
    
            TxN1=n3*t2-n2*t3
            TxN2=n1*t3-n3*t1
            TxN3=n2*t1-n1*t2
    
            BT=(B1*t1+B2*t2+B3*t3)
    
            gu1sq=dsqrt(1.d0/gam_d)
    
            lu(1,1)   = 1.d0/(sqgam*rhosq)
            lu(1,2)   = 0.d0
            lu(1,3)   = 0.d0
            lu(1,4)   = 0.d0
            lu(1,5)   = -1.d0/(a2*sqgam*rhosq)
            lu(1,6)   = 0.d0
            lu(1,7)   = 0.d0
            lu(1,8)   = 0.d0
    
            nen = (n3**2*(t1**2+t2**2)-2.d0*n1*n3*t1*t3-2.d0*n2*t2*       &
                    (n1*t1+n3*t3)+n2**2*(t1**2+t3**2)+n1**2*(t2**2+t3**2))
    
            nen2=a*gu1sq*nen
    
            lu(2,1)   = 0.d0
            lu(2,2)   = 0.d0
            lu(2,3)   = 0.d0
            lu(2,4)   = 0.d0
            lu(2,5)   = 0.d0
            lu(2,6)   = (-n2*t1*t2-n3*t1*t3+n1*(t2**2+t3**2))/nen2
            lu(2,7)   = (-t2*(n1*t1+n3*t3)+n2*(t1**2+t3**2))/nen2
            lu(2,8)   = (n3*(t1**2+t2**2)-(n1*t1+n2*t2)*t3)/nen2
    
            nen3 = sqrt(2.d0)*sqrt(abs(p))*nen
            nen31 = nen3/sqrt(rho)
    
            lu(3,1)   = 0.d0
            lu(3,2)   = rho*(-TxN1)/nen3
            lu(3,3)   = rho*(-TxN2)/nen3
            lu(3,4)   = rho*(-TxN3)/nen3
            lu(3,5)   = 0.d0
            lu(3,6)   = TxN1/nen31
            lu(3,7)   = TxN2/nen31
            lu(3,8)   = TxN3/nen31
    
            lu(4,1)   = lu(3,1)
            lu(4,2)   = -lu(3,2)
            lu(4,3)   = -lu(3,3)
            lu(4,4)   = -lu(3,4)
            lu(4,5)   = lu(3,5)
            lu(4,6)   = lu(3,6)
            lu(4,7)   = lu(3,7)
            lu(4,8)   = lu(3,8)
    
            Term51 = rho*cf*((n2*t1*t2+n3*t1*t3-n1*(t2**2+t3**2))*rhosq*cf**2* &
                    alphaf-a*BN*alphas*(-n2**2*t1+n1*n2*t2+n3*TxN2))
    
            Term52 = rho*cf*((-t2*(n1*t1+n3*t3)+n2*(t1**2+t3**2))*rhosq*cf**2 &
                    *alphaf-a*BN*alphas*(-n1*n2*t1+n1**2*t2+n3*TxN1))
    
            Term53 = rho*cf*((n3*(t1**2+t2**2)-(n1*t1+n2*t2)*t3)*rhosq*cf**2* &
                    alphaf-a*(-n1*n3*t1+n1**2*t3+n2*(-n3*t2+n2*t3))*BN*alphas)
    
            Term54 = alphaf/(twosq*a**2*gu1sq*rhosq*(alphaf**2+alphas**2))
    
            Term55 = (n2**2*t1-n1*n2*t2+n3*(n3*t1-n1*t3))*alphas
    
            Term56 = alphas*(-n1*n2*t1+n1**2*t2+n3*TxN1)
            Term57 = (-n1*n3*t1+n1**2*t3+n2*(-n3*t2+n2*t3))*alphas
    
            nen51 = twosq*a*nen*gu1sq*(a*BN**2*alphas**2+rhosq*cf**2*alphaf*  &
                    (a*rhosq*alphaf+BT*alphas))
    
            nen52 = twosq*a*gu1sq*(alphaf**2+alphas**2)*nen
    
            lu(5,1)  = 0.d0
            lu(5,2)  = -Term51/nen51
            lu(5,3)  = Term52/nen51
            lu(5,4)  = Term53/nen51
            lu(5,5)  = Term54
            lu(5,6)  = Term55/nen52
            lu(5,7)  = Term56/nen52
            lu(5,8)  = Term57/nen52

            lu(6,1)  = lu(5,1)
            lu(6,2)  = -lu(5,2)
            lu(6,3)  = -lu(5,3)
            lu(6,4)  = -lu(5,4)
            lu(6,5)  = lu(5,5)
            lu(6,6)  = lu(5,6)
            lu(6,7)  = lu(5,7)
            lu(6,8)  = lu(5,8)
 
            Term71 = rho*cf*(a*(n2**2*t1-n1*n2*t2+n3*(n3*t1-n1*t3))*rhosq*  &
                    alphaf+((B3*n2*t1-B2*n3*t1-B3*n1*t2+B2*n1*t3)*(-n3*t2  &
                    +n2*t3)+B1*(n2**2*t1**2+n3**2*t1**2-2.d0*n1*n2*t1*t2-2.d0 &
                    *n1*n3*t1*t3+n1**2*(t2**2+t3**2)))*alphas)
    
            Term72 = rho*cf*(((n3*t1-n1*t3)*(B3*n2*t1-B3*n1*t2+B1*n3*t2-B1  &
                    *n2*t3)+B2*((n1**2+n3**2)*t2**2-2.d0*n2*t2*(n1*t1+n3*t3)  &
                    +n2**2*(t1**2+t3**2)))*alphas+a*rhosq*alphaf*(-n1*n2*t1   &
                    +n1**2*t2+n3*TxN1))
    
            Term73 = rho*cf*(a*(-n1*n3*t1+n1**2*t3+n2*(-n3*t2+n2*t3))*rhosq &
                    *alphaf+alphas*(B3*(n3**2*(t1**2+t2**2)-2.d0*n3*      &
                    (n1*t1+n2*t2)*t3+(n1**2+n2**2)*t3**2)+B2*             &
                    (n3*t1-n1*t3)*TxN3+B1*(-n3*t2+n2*t3)*TxN3))
    
            Term74 = alphas/(twosq*a**2*gu1sq*rhosq*(alphaf**2+alphas**2))
    
            Term75 = -alphaf*(n2**2*t1-n1*n2*t2+n3*(n3*t1-n1*t3))
    
            Term76 = -alphaf*(-n1*n2*t1+n1**2*t2+n3*TxN1)
    
            Term77 = -alphaf*(-n1*n3*t1+n1**2*t3+n2*(-n3*t2+n2*t3))
    
    
            nen71 = twosq*beta1*nen                                         &
                    *gu1sq*(a*BN**2*alphas**2+rhosq*cf**2*alphaf*(a*rhosq*  &
                    alphaf+BT*alphas))
    
            nen72 = nen52
    
            lu(7,1)= 0.d0
            lu(7,2)= Term71/nen71
            lu(7,3)= Term72/nen71
            lu(7,4)= Term73/nen71
            lu(7,5)= Term74
            lu(7,6)= Term75/nen72
            lu(7,7)= Term76/nen72
            lu(7,8)= Term77/nen72
    

            lu(8,1)=lu(7,1)
            lu(8,2)=-lu(7,2)
            lu(8,3)=-lu(7,3)
            lu(8,4)=-lu(7,4)
            lu(8,5)=lu(7,5)
            lu(8,6)=lu(7,6)
            lu(8,7)=lu(7,7)
            lu(8,8)=lu(7,8)
    
            do m=1,8
                evl(m,1) =lu(m,1)*g1-lu(m,2)*u1*g1/rho-lu(m,3)*u2*g1/rho-   &
                        lu(m,4)*u3*g1/rho+lu(m,5)*g1**2*(u1**2+u2**2+u3**2)*.5d0
                evl(m,2) =-lu(m,5)*u1*g1**2+lu(m,2)*g1/rho
                evl(m,3) =-lu(m,5)*u2*g1**2+lu(m,3)*g1/rho
                evl(m,4) =-lu(m,5)*u3*g1**2+lu(m,4)*g1/rho
                evl(m,5) =lu(m,5)*g1**2
                evl(m,6) =lu(m,6)*g1-B1*lu(m,5)*g1**2
                evl(m,7) =lu(m,7)*g1-B2*lu(m,5)*g1**2
                evl(m,8) =lu(m,8)*g1-B3*lu(m,5)*g1**2
            enddo
    
    
        end subroutine compute_evl_gpu
    
    attributes(device) subroutine reconstruct_gpu(u_stencil,Rmat,Rinv,uf)
        use setup_device,only:nVar_d
        implicit none
        double precision :: u_stencil(8,5),Rmat(8,8),Rinv(8,8),flux(8),&
                            v_stencil(8,5),vp0(8),vp1(8),vp2(8),vf(8),tau5
        double precision:: beta0, beta1, beta2,&
                            omega0, omega1, omega2, omega_sum
        double precision,intent(out) :: uf(8)
        double precision::tol = 1d-5
        integer :: i, d0, d1, d2
    
        do i = 1,5
            v_stencil(:,i) = matmul(Rinv(:,:),u_stencil(:,i))
        end do
    
        vp0 = 1.d0/3*v_stencil(:,1)-7.d0/6*v_stencil(:,2)+11.d0/6*v_stencil(:,3)
        vp1 = -1.d0/6*v_stencil(:,2)+5.d0/6*v_stencil(:,3)+1.d0/3*v_stencil(:,4)
        vp2 = 1.d0/3*v_stencil(:,3)+5.d0/6*v_stencil(:,4)-1.d0/6*v_stencil(:,5)
    
        do i = 1,nVar_d
            beta0 = 13.d0/12*&
                    (v_stencil(i,1)-2.d0*v_stencil(i,2)+v_stencil(i,3))**2.d0+&
                    1.d0/4*(v_stencil(i,1)-4.d0*v_stencil(i,2)+3.d0*v_stencil(i,3))**2.d0
    
            beta1 = 13.d0/12*&
                    (v_stencil(i,2)-2.d0*v_stencil(i,3)+v_stencil(i,4))**2.d0+&
                    1.d0/4*(v_stencil(i,2)-v_stencil(i,4))**2.d0
    
            beta2 = 13.d0/12*&
                    (v_stencil(i,3)-2.d0*v_stencil(i,4)+v_stencil(i,5))**2.d0+&
                    1.d0/4*(3.d0*v_stencil(i,3)-4.d0*v_stencil(i,4)+v_stencil(i,5))**2.d0
    
            tau5 = dabs(beta0-beta2)
            omega0 = (1.d0 + tau5/(beta0+1d-40))**6.d0
            omega1 = (1.d0 + tau5/(beta1+1d-40))**6.d0
            omega2 = (1.d0 + tau5/(beta2+1d-40))**6.d0
    
            omega_sum = omega0 + omega1 + omega2
            omega0 = omega0/omega_sum
            omega1 = omega1/omega_sum
            omega2 = omega2/omega_sum
    
    
            if (omega0 < tol) then
                d0 = 0
            else
                d0 = 1
            end if
    
            if (omega1 < tol) then
                d1 = 0
            else
                d1 = 1
            end if
    
            if (omega2 < tol) then
                d2 = 0
            else
                d2 = 1
            end if
    
            if (d0==1 .and. d1==1 .and. d2==1)then
                vf(i) = 0.1d0*vp0(i) + 0.6d0*vp1(i) + 0.3d0*vp2(i)
            elseif(d0==1 .and. d1==1)then
                vf(i) = 0.25d0*vp0(i) + 0.75d0*vp1(i)
            elseif(d1==1 .and. d2==1)then
                vf(i) = 0.5d0*vp1(i) + 0.5d0*vp2(i)
            elseif(d0==1)then
                vf(i) = vp0(i)
            elseif(d1==1)then
                vf(i) = vp1(i)
            elseif(d2==1)then
                vf(i) = vp2(i)
            else
                write(*,*)'Error.'
                stop
            end if
    
        end do
    
        uf = matmul(Rmat,vf)
    
        
    end subroutine reconstruct_gpu
    
    attributes(global) subroutine calDivB_gpu()
            use setup_device, only: hx_d, hy_d, ub_d, dB_d, npx_d, npy_d,&
            neighbx_d,neighby_d,uInlet_d
            implicit none
            integer::i, j, pt_index, index
            double precision:: tmp, var(5)
    
            i=threadidx%x+(blockidx%x-1)*blockdim%x
            j=threadidx%y+(blockidx%y-1)*blockdim%y
            if(i<=npx_d .and. j<=npy_d)then
                do index=1,5
                    if(neighbx_d(i,index)==0)then
                        var(index) = uInlet_d(6)
                    else
                        var(index) = ub_d(neighbx_d(i,index),j,6)
                    endif
                enddo
                tmp = 1.d0/(12.d0*hx_d)*(var(1) - 8.d0*var(2) + 8.d0*var(4) - var(5))
    
                do index=1,5
                    if(neighby_d(j,index)==0)then
                        var(index) = uInlet_d(7)
                    else
                        var(index) = ub_d(i,neighby_d(j,index),7)
                    endif
                enddo
                tmp = tmp + 1.d0/(12.d0*hy_d)*(var(1) - 8.d0*var(2) + 8.d0*var(4) - var(5))
    
                dB_d(i,j) = tmp
    
            endif
    
        end subroutine calDivB_gpu
    
        attributes(global) subroutine ComputeSources_GD_gpu
            use setup_device
            implicit none
            integer::i, j, pt_index, index
            double precision:: tmp(8), var(5)
            double precision:: divB, rho, u1, u2, u3, B1, B2, B3
            double precision:: dLossdBx, dLossdBy, res
    
            i=threadidx%x+(blockidx%x-1)*blockdim%x
            j=threadidx%y+(blockidx%y-1)*blockdim%y
            if(i<=npx_d .and. j<=npy_d)then
                dLossdBx=0.d0
                dLossdBy=0.d0

                do index=1,5
                    if(neighbx_d(i,index)==0)then
                        if(i-3+index<1)then
                            var(index) = dB_d(1,j)
                        else
                            var(index) = dB_d(npx_d,j)
                        endif
                    else
                        var(index) = dB_d(neighbx_d(i,index),j)
                    endif
                enddo
                dLossdBx = dLossdBx + 2.d0*var(1)/(12.d0*hx_d)
                dLossdBx = dLossdBx - 2.d0*var(2)*2.d0/(3.d0*hx_d)
                dLossdBx = dLossdBx + 2.d0*var(4)*2.d0/(3.d0*hx_d)
                dLossdBx = dLossdBx - 2.d0*var(5)/(12.d0*hx_d)
 
                do index=1,5
                    if(neighby_d(j,index)==0)then
                        if(j-3+index<1)then
                            var(index) = dB_d(i,1)
                        else
                            var(index) = dB_d(i,npy_d)
                        endif
                    else
                        var(index) = dB_d(i,neighby_d(j,index))
                    endif
                enddo
                dLossdBy = dLossdBy + 2.d0*var(1)/(12.d0*hy_d)
                dLossdBy = dLossdBy - 2.d0*var(2)*2.d0/(3.d0*hy_d)
                dLossdBy = dLossdBy + 2.d0*var(4)*2.d0/(3.d0*hy_d)
                dLossdBy = dLossdBy - 2.d0*var(5)/(12.d0*hy_d)
    
                divB = dB_d(i,j)
                rho = ub_d(i,j,1)
                u1 = ub_d(i,j,2)/rho
                u2 = ub_d(i,j,3)/rho
                u3 = ub_d(i,j,4)/rho
                B1 = ub_d(i,j,6)
                B2 = ub_d(i,j,7)
                B3 = ub_d(i,j,8)
    
                tmp(1) = 0.d0
                tmp(2) = divB*B1
                tmp(3) = divB*B2
                tmp(4) = divB*B3
                tmp(5) = divB*(u1*B1+u2*B2+u3*B3)
                tmp(6) = divB*u1
                tmp(7) = divB*u2
                tmp(8) = divB*u3
    
                tmp(6) = tmp(6) + dLossdBx*eta_d
                tmp(7) = tmp(7) + dLossdBy*eta_d
    
                resid_d(i,j,:) = resid_d(i,j,:) - tmp(:)
    
            endif
    
        end subroutine ComputeSources_GD_gpu
    
    attributes(global) subroutine ssprk_stage_1()
        use setup_device
        implicit none
        integer::i,j
        double precision, parameter :: b10=0.377268915331368d0
    
        i=threadidx%x+(blockidx%x-1)*blockdim%x
        j=threadidx%y+(blockidx%y-1)*blockdim%y
        if(i<=npx_d .and. j<=npy_d)then
    
            ub_d(i,j,:) = ub_d(i,j,:)+b10*dt_d*resid_d(i,j,:)
    
        endif
        
    end subroutine ssprk_stage_1
    
    attributes(global) subroutine ssprk_stage_2()
        use setup_device
        implicit none
        integer::i,j
        double precision, parameter :: b21=0.377268915331368d0
    
        i=threadidx%x+(blockidx%x-1)*blockdim%x
        j=threadidx%y+(blockidx%y-1)*blockdim%y
        if(i<=npx_d .and. j<=npy_d)then
    
            ub_d(i,j,:) = ub_d(i,j,:)+b21*dt_d*resid_d(i,j,:)

        endif
    
    end subroutine ssprk_stage_2
    
    attributes(global) subroutine ssprk_stage_3()
        use setup_device
        implicit none
        integer::i,j
        double precision, parameter :: a30=0.355909775063327d0
        double precision, parameter :: a32=0.644090224936674d0
        double precision, parameter :: b32=0.242995220537396d0
    
        i=threadidx%x+(blockidx%x-1)*blockdim%x
        j=threadidx%y+(blockidx%y-1)*blockdim%y
        if(i<=npx_d .and. j<=npy_d)then
    
            ub_d(i,j,:) = a30*ub0_d(i,j,:) + a32*ub_d(i,j,:) + b32*dt_d*resid_d(i,j,:)
    
        endif
        
    end subroutine ssprk_stage_3
    
    attributes(global) subroutine ssprk_stage_4()
        use setup_device
        implicit none
        integer::i,j
        double precision, parameter :: a40=0.367933791638137d0
        double precision, parameter :: a43=0.632066208361863d0
        double precision, parameter :: b43=0.238458932846290d0
    
        i=threadidx%x+(blockidx%x-1)*blockdim%x
        j=threadidx%y+(blockidx%y-1)*blockdim%y
        if(i<=npx_d .and. j<=npy_d)then
    
            ub_d(i,j,:) = a40*ub0_d(i,j,:) + a43*ub_d(i,j,:) + b43*dt_d*resid_d(i,j,:)

        endif
        
    end subroutine ssprk_stage_4
    
    attributes(global) subroutine ssprk_stage_5()
        use setup_device
        implicit none
        integer::i,j
        double precision, parameter :: a52=0.237593836598569d0
        double precision, parameter :: a54=0.762406163401431d0
        double precision, parameter :: b54=0.287632146308408d0
        
        i=threadidx%x+(blockidx%x-1)*blockdim%x
        j=threadidx%y+(blockidx%y-1)*blockdim%y
        if(i<=npx_d .and. j<=npy_d)then
    
            ub_d(i,j,:) = a52*uba_d(i,j,:) + a54*ub_d(i,j,:) + b54*dt_d*resid_d(i,j,:)

        endif
    
    end subroutine ssprk_stage_5
    
endmodule kernel_function

program MHD
    use setup
    use setup_device
    use kernel_function
    use, intrinsic::ieee_arithmetic
    implicit none

    integer :: ipx,ipy,iBoundary,argc,iostat
    double precision :: x,y
    double precision :: divB_global = 0.d0, divB_tmp = 0.0d0
    double precision :: rho,u,v,w,p,Bx,By,Bz
    logical:: is_nan
    character*64 :: filename, command, full_path, casename,inputFile

    call cpu_time(timeStart)
    
    argc=command_argument_count()
    if(argc==0)then
        inputFile="input.dat"
    elseif(argc==1)then
        call get_command_argument(1,inputFile)
    else
        Write(*,*)"Wrong parameters"
    endif

    call date_and_time(values=date_values)

    open(11,file=inputFile,STATUS="OLD", ACTION="READ",IOSTAT=iostat)

    if (iostat /= 0) then
        PRINT *, "Error opening input file! It may not exist."
        stop
    else
        PRINT *, "Input file opened successfully."
    endif

    read(11,*)
    read(11,*)casename
    read(11,*)
    read(11,*)xmin,xmax,ymin,ymax,npx,npy
    read(11,*)
    read(11,*)dt,tf,NTF,eta,gam
    read(11,*)
    read(11,*)boundary(1),boundary(2),boundary(3),boundary(4)
    read(11,*)
    read(11,*)rho,u,v,w,p,Bx,By,Bz
    read(11,*)
    read(11,*)restart
    if(restart==1)then
        read(11,'(A)')namerestart
        namerestart = trim(namerestart)
    endif
    close(11)

    casename = trim(casename)
    call ToUpper(casename)
    if(casename=="ORSZAGTANG")then
        casename = "OT"
    elseif(casename=="CLOUDSHOCK1")then
        casename = "CS1"
    elseif(casename=="CLOUDSHOCK2")then
        casename = "CS2"
    elseif(casename=="ROTOR1")then
        casename = "Rotor1"
    elseif(casename=="ROTOR2")then
        casename = "Rotor2"
    elseif(casename=="KELVIN-HELMHOLTZ")then
        casename = "KH"
    else
        write(*,*)"Custom test cases are currently not &
        supported through the input.dat file and require&
        modifications to the initialization function&
        in the source code."
        write(*,*)casename
        stop
    endif


    !iBoundary: 1-xmin, 2-xmax, 3-ymin, 4-ymax
    do iBoundary=1,4
        boundary(iBoundary) = trim(boundary(iBoundary))
        if(boundary(iBoundary)=='inlet')then
            bound_type(iBoundary) = 1
        elseif(boundary(iBoundary)=='out')then
            bound_type(iBoundary) = 2
        elseif(boundary(iBoundary)=='cyc')then
            bound_type(iBoundary) = 3
        elseif(boundary(iBoundary)=='sym')then
            bound_type(iBoundary) = 4
        else
            write(*,*)'Wrong boundary type of boundary ', iBoundary
        endif
    enddo

    if(bound_type(1)==1.or.&
       bound_type(2)==1.or.&
       bound_type(3)==1.or.&
       bound_type(4)==1)then
        uInlet(1) = rho
        uInlet(2) = rho*u
        uInlet(3) = rho*v
        uInlet(4) = rho*w
        uInlet(5) = p/(gam-1.d0)+0.5*rho*(u**2+v**2+w**2)+0.5*(Bx**2+By**2+Bz**2)
        uInlet(6) = Bx
        uInlet(7) = By
        uInlet(8) = Bz
    endif

    write(folder_path, '(A,"_",I2.2,".",I2.2,".",I2.2,".",I2.2)') &
        trim(casename),date_values(2), date_values(3), date_values(5), date_values(6)

    write(command, '(A," ",A)') 'mkdir -p', trim(folder_path)
    call execute_command_line(command, wait=.true., exitstat=ierr)

    if (ierr /= 0) then
        print *, "Error creating folder: ", trim(folder_path)
        stop
    end if

    hx = (xmax-xmin)/dble(npx-1)
    hy = (ymax-ymin)/dble(npy-1)
    maxiter = nint(tf/dt)

    allocate(ub(npx,npy,nVar)) !conserved variables
    allocate(resid(0:npx+1,0:npy+1,nVar))
    allocate(neighbxp(npx+1,5))
    allocate(neighbxm(npx+1,5))
    allocate(neighbyp(npy+1,5))
    allocate(neighbym(npy+1,5))
    allocate(neighbx(npx+1,5))
    allocate(neighby(npy+1,5))

   
    if (restart==0) then
        call ToUpper(casename)
        if(casename=="OT")then
            call orszagInitialize
        elseif(casename=="CS1")then
            call CloudShock1Initailize
        elseif(casename=="CS2")then
            call CloudShock2Initailize
        elseif(casename=="ROTOR1")then
            call Rotor1Initialize
        elseif(casename=="ROTOR2")then
            call Rotor2Initialize
        elseif(casename=="KH")then
            call KH_Initialize
        else
            write(*,*)"Custom test cases are currently not &
            supported through the input.dat file and require&
            modifications to the initialization function&
            in the source code."
            write(*,*)casename
            stop
        endif
        
        ctime = 1d-10
        iter = 0
        call write_solu_tecplot
    else if (restart==1) then
        call read_data
    end if

    call boundary_condition_cpu

    filename = "Info.dat"
    write(full_path,'(A,"/",A)') trim(folder_path), trim(filename)
    open(333,file=full_path)
    write(333,*)trim(casename),'npx = ',npx,',npy = ',npy,',dt = ', dt,&
     'eta = ', eta,' NTF=',NTF
    close(333)

    write(*,*)'allocating '
    call gpu_allocate
    write(*,*)'allocate end'
    residglob=0.d0
    call cpu_time(timeStart1)
    do while (ctime<=tf)
        
        call SSPRK_time_stepping_gpu

        call cpu_time(timeStart2)

        iter = iter+1

        if(iter==1 .or. mod(iter,50)==0 .or.iter==maxiter)then

            resid = resid_d
            residglob = norm2(resid(1:npx,1:npy,1:nVar))/dble(4*npx*npy)
            is_nan = ieee_is_nan(residglob)
            if(is_nan)then
                write(*,*)'Error.'
                stop
            end if

        endif

        if(iter==1 .or.mod(iter,10)==0 .or.iter==maxiter)then
            print *,'iter=',iter,'time=',ctime,'resid=',residglob         
        endif

        if (mod(iter,NTF)==0.or.iter==maxiter) then

            call cpu_time(timeStop1)

            ub=ub_d

            call write_solu_tecplot
            call write_data
            call cpu_time(timeStart1)
        end if
    end do
    call cpu_time(timeEnd)
    print *, 'Total time = ',timeEnd-timeStart,'s'

end program MHD

SUBROUTINE ToUpper(s)
    CHARACTER(*) :: s
    INTEGER :: i
    DO i = 1, LEN_TRIM(s)
        SELECT CASE (s(i:i))
            CASE ('a':'z')
                s(i:i) = CHAR(ICHAR(s(i:i)) - 32)
        END SELECT
    END DO
END SUBROUTINE ToUpper

subroutine gpu_allocate
    use setup
    use setup_device
    implicit none

    gam_d = gam
    npx_d = npx
    npy_d = npy
    nVar_d = nVar
    dt_d = dt
    hx_d = hx
    hy_d = hy
    eta_d = eta
    bound_type_d=bound_type

    allocate(ub_d(npx,npy,nVar),stat=ierr); !call gerror_report(ierr)
    allocate(ub0_d(npx,npy,nVar),stat=ierr); !call gerror_report(ierr)
    allocate(uba_d(npx,npy,nVar),stat=ierr); !call gerror_report(ierr)
    ub_d = ub
    allocate(resid_d(0:npx+1,0:npy+1,nVar), stat=ierr); !call gerror_report(ierr)
    resid_d = resid
    allocate(dB_d(1:npx,1:npy),stat=ierr)
    allocate(neighbxp_d(npx+1,5))
    allocate(neighbxm_d(npx+1,5))
    allocate(neighbyp_d(npy+1,5))
    allocate(neighbym_d(npy+1,5))
    allocate(neighbx_d(npx+1,5))
    allocate(neighby_d(npy+1,5))
    neighbxp_d = neighbxp
    neighbxm_d = neighbxm
    neighbyp_d = neighbyp
    neighbym_d = neighbym
    neighbx_d = neighbx
    neighby_d = neighby
    allocate(uInlet_d(8))
    uInlet_d=uInlet

end subroutine gpu_allocate

subroutine orszagInitialize
    use setup, only:gam,ub,npx,npy,hx,hy
    implicit none
    integer::ipx,ipy
    double precision:: x, y, uc(8)
    double precision:: rho, u1, u2, u3, B1, B2, B3, pr

    do ipy = 1,npy
        do ipx = 1,npx

            x = dble(ipx-1)*hx
            y = dble(ipy-1)*hy

            rho = gam*gam
            u1 = -sin(y)
            u2 = sin(x)
            u3 = 0.d0
            B1 = -sin(y)
            B2 = sin(2.d0*x)
            B3 = 0.d0
            pr = gam

            ub(ipx,ipy,1) = gam*gam
            ub(ipx,ipy,2) = rho*u1
            ub(ipx,ipy,3) = rho*u2
            ub(ipx,ipy,4) = rho*u3
            ub(ipx,ipy,5) = pr/(gam-1.d0)+0.5*rho*(u1**2+u2**2+u3**2)+0.5*(B1**2+B2**2+B3**2)
            ub(ipx,ipy,6) = B1
            ub(ipx,ipy,7) = B2
            ub(ipx,ipy,8) = B3
            
        end do
    end do

    

end subroutine orszagInitialize

subroutine CloudShock1Initailize
    use setup
    implicit none
    integer::ipx, ipy
    double precision::x,y,r
    double precision:: rho, u1, u2, u3, pr, B1, B2, B3

    do ipy = 1, npy
        do ipx = 1,npx
            x = dble(ipx-1)*hx
            y = dble(ipy-1)*hy
            !r = dsqrt((x-0.5)**2.d0+(y-0.5)**2.d0)
            !fr = (r1-r)/(r1-r0)
            if (x < 0.6d0)then
                rho = 3.86859d0
                u1 = 0.d0
                u2 = 0.d0
                u3 = 0.d0
                pr = 167.345d0
                B1 = 0.d0
                B2 = 7.73718d0
                B3 = -7.73718d0

            else
                r = dsqrt((x-0.8d0)**2.d0+(y-0.5d0)**2.d0)
                if (r < 0.15)then
                    rho = 10.d0
                    u1 = -11.2536d0
                    u2 = 0.d0
                    u3 = 0.d0
                    pr = 1.d0
                    B1 = 0.d0
                    B2 = 2.d0
                    B3 = 2.d0
                else
                    rho = 1.d0
                    u1 = -11.2536d0
                    u2 = 0.d0
                    u3 = 0.d0
                    pr = 1.d0
                    B1 = 0.d0
                    B2 = 2.d0
                    B3 = 2.d0
                end if
            end if

            ub(ipx,ipy,1) = rho
            ub(ipx,ipy,2) = rho*u1
            ub(ipx,ipy,3) = rho*u2
            ub(ipx,ipy,4) = rho*u3
            ub(ipx,ipy,5) = pr/(gam-1.d0)+0.5*rho*(u1**2+u2**2+u3**2)+0.5*(B1**2+B2**2+B3**2)
            ub(ipx,ipy,6) = B1
            ub(ipx,ipy,7) = B2
            ub(ipx,ipy,8) = B3

        end do
    end do

end subroutine

subroutine CloudShock2Initailize
    use setup
    implicit none
    integer::ipx, ipy
    double precision::x,y,r
    double precision:: rho, u1, u2, u3, pr, B1, B2, B3

    do ipy = 1, npy
        do ipx = 1,npx
            x = dble(ipx-1)*hx
            y = dble(ipy-1)*hy
            !r = dsqrt((x-0.5)**2.d0+(y-0.5)**2.d0)
            !fr = (r1-r)/(r1-r0)
            if (x < 0.05)then
                rho = 3.86859d0
                u1 = 11.2536d0
                u2 = 0.d0
                u3 = 0.d0
                pr = 167.345d0
                B1 = 0.d0
                B2 = 2.1826182d0
                B3 = -2.1826182d0
            else
                r = dsqrt((x-0.25d0)**2.d0+(y-0.5d0)**2.d0)
                if (r < 0.15)then
                    rho = 10.d0
                    u1 = 0.d0
                    u2 = 0.d0
                    u3 = 0.d0
                    pr = 1.d0
                    B1 = 0.d0
                    B2 = 0.56418958d0
                    B3 = 0.56418958d0
                else
                    rho = 1.d0
                    u1 = 0.d0
                    u2 = 0.d0
                    u3 = 0.d0
                    pr = 1.d0
                    B1 = 0.d0
                    B2 = 0.56418958d0
                    B3 = 0.56418958d0
                end if
            end if

            ub(ipx,ipy,1) = rho
            ub(ipx,ipy,2) = rho*u1
            ub(ipx,ipy,3) = rho*u2
            ub(ipx,ipy,4) = rho*u3
            ub(ipx,ipy,5) = pr/(gam-1.d0)+0.5*rho*(u1**2+u2**2+u3**2)+0.5*(B1**2+B2**2+B3**2)
            ub(ipx,ipy,6) = B1
            ub(ipx,ipy,7) = B2
            ub(ipx,ipy,8) = B3

        end do
    end do

end subroutine

subroutine Rotor1Initialize
    use setup
    implicit none
    integer::ipx, ipy
    double precision::x,y,r,fr
    double precision:: rho, u1, u2, u3
    double precision:: B1, B2 = 1d-6, B3 = 1d-6, pr = 1.d0
    double precision:: u0 = 2.d0, r0 = 0.1, r1 = 0.115

    B1 = 5.d0/dsqrt(4.d0*PI)

    do ipy = 1, npy
        do ipx = 1,npx
            x = dble(ipx-1)*hx
            y = dble(ipy-1)*hy
            r = dsqrt((x-0.5)**2.d0+(y-0.5)**2.d0)
            fr = (r1-r)/(r1-r0)
            if (r<=r0) then
                rho = 10
                u1 = -u0*(y-0.5)/r0
                u2 = u0*(x-0.5)/r0
                u3 = 0.d0
            elseif (r>r0 .and. r<=r1) then
                rho = 1.d0 + 9.d0*fr
                u1 = -u0*(y-0.5)*fr/r0
                u2 = u0*(x-0.5)*fr/r0
                u3 = 0.d0
            else
                rho = 1.d0
                u1 = 0.d0
                u2 = 0.d0
                u3 = 0.d0
            end if

            ub(ipx,ipy,1) = rho
            ub(ipx,ipy,2) = rho*u1
            ub(ipx,ipy,3) = rho*u2
            ub(ipx,ipy,4) = rho*u3
            ub(ipx,ipy,5) = pr/(gam-1.d0)+0.5*rho*(u1**2+u2**2+u3**2)+0.5*(B1**2+B2**2+B3**2)
            ub(ipx,ipy,6) = B1
            ub(ipx,ipy,7) = B2
            ub(ipx,ipy,8) = B3

        end do
    end do

end subroutine Rotor1Initialize

subroutine Rotor2Initialize
    use setup
    implicit none
    integer::ipx, ipy
    double precision::x,y,r,fr
    double precision:: rho, u1, u2, u3
    double precision:: B1, B2 = 1d-40, B3 = 1d-40, pr = 0.5d0
    double precision:: u0 = 1.d0, r0 = 0.1, r1 = 0.115

    B1 = 2.5d0/dsqrt(4.d0*PI)

    do ipy = 1, npy
        do ipx = 1,npx
            x = dble(ipx-1)*hx
            y = dble(ipy-1)*hy
            r = dsqrt((x-0.5d0)**2.d0+(y-0.5d0)**2.d0)
            fr = (r1-r)/(r1-r0)
            if (r<=r0) then
                rho = 10d0
                u1 = -u0*(y-0.5d0)/r0
                u2 = u0*(x-0.5d0)/r0
                u3 = 0.d0
            elseif (r>r0 .and. r<=r1) then
                rho = 1.d0 + 9.d0*fr
                u1 = -u0*(y-0.5d0)*fr/r0
                u2 = u0*(x-0.5d0)*fr/r0
                u3 = 0.d0
            else
                rho = 1.d0
                u1 = 0.d0
                u2 = 0.d0
                u3 = 0.d0
            end if

            ub(ipx,ipy,1) = rho
            ub(ipx,ipy,2) = rho*u1
            ub(ipx,ipy,3) = rho*u2
            ub(ipx,ipy,4) = rho*u3
            ub(ipx,ipy,5) = pr/(gam-1.d0)+0.5*rho*(u1**2+u2**2+u3**2)+0.5*(B1**2+B2**2+B3**2)
            ub(ipx,ipy,6) = B1
            ub(ipx,ipy,7) = B2
            ub(ipx,ipy,8) = B3

        end do
    end do

end subroutine Rotor2Initialize

subroutine KH_Initialize
    use setup
    implicit none
    integer::ipx, ipy
    double precision::x,y
    double precision:: rho=1.d0, u1, u2, u3, pr, B1, B2, B3
    double precision:: M=1.d0, y0=1.d0/20, Ca = 0.1d0, theta

    theta = PI/3.d0
    pr=1.d0/gam
    do ipx=1,npx
        do ipy=1,npy
            x = dble(ipx-1)*hx
            y = -1.d0 + dble(ipy-1)*hy

            u1 = M/2.d0*dtanh(y/y0)
            u2 = 0.01d0*dsin(2*PI*x)*exp(-y**2.d0/0.01)
            u3 = 0.d0
            B1 = Ca*sqrt(rho)*dcos(theta)
            B2 = 0.d0
            B3 = Ca*sqrt(rho)*dsin(theta)

            ub(ipx,ipy,1) = rho
            ub(ipx,ipy,2) = rho*u1
            ub(ipx,ipy,3) = rho*u2
            ub(ipx,ipy,4) = rho*u3
            ub(ipx,ipy,5) = pr/(gam-1.d0)+0.5*rho*(u1**2+u2**2+u3**2)+0.5*(B1**2+B2**2+B3**2)
            ub(ipx,ipy,6) = B1
            ub(ipx,ipy,7) = B2
            ub(ipx,ipy,8) = B3

        end do
    end do

end subroutine KH_Initialize

subroutine boundary_condition_cpu
    use setup
    implicit none
    integer::i,ib,ipx,ipy,pt_index,ptp_index,ptm_index,tmp

    do ib=1,4

        if(ib==1.or.ib==2)then!x-direction
            !face
            do ipx=1,npx+1
                !f+
                do i=1,5
                    pt_index=ipx-4+i
                    if(pt_index.lt.1)then
                        if(bound_type(ib)==1)then!inlet
                            neighbxp(ipx,i) = 0
                        elseif(bound_type(ib)==2)then!outflow
                            neighbxp(ipx,i) = 1
                        elseif(bound_type(ib)==3)then!cyclic
                            neighbxp(ipx,i) = npx+pt_index-1
                        elseif(bound_type(ib)==4)then!sym
                            neighbxp(ipx,i) = 2-pt_index
                        else
                            write(*,*)'Wrong boundary type'
                            stop
                        endif
                    elseif(pt_index.gt.npx)then
                        if(bound_type(ib)==1)then!inlet
                            neighbxp(ipx,i) = 0
                        elseif(bound_type(ib)==2)then!outflow
                            neighbxp(ipx,i) = npx
                        elseif(bound_type(ib)==3)then!cyclic
                            neighbxp(ipx,i) = pt_index-npx+1
                        elseif(bound_type(ib)==4)then!sym
                            neighbxp(ipx,i) = 2*npx-pt_index
                        else
                            write(*,*)'Wrong boundary type'
                            stop
                        endif
                    else
                        neighbxp(ipx,i)=pt_index
                    endif
                enddo
                !f-
                do i=1,5
                    pt_index=ipx+3-i
                    if(pt_index.lt.1)then
                        if(bound_type(ib)==1)then!inlet
                            neighbxm(ipx,i) = 0
                        elseif(bound_type(ib)==2)then!outflow
                            neighbxm(ipx,i) = 1
                        elseif(bound_type(ib)==3)then!cyclic
                            neighbxm(ipx,i) = npx+pt_index-1
                        elseif(bound_type(ib)==4)then!sym
                            neighbxm(ipx,i) = 2-pt_index
                        else
                            write(*,*)'Wrong boundary type'
                            stop
                        endif
                    elseif(pt_index.gt.npx)then
                        if(bound_type(ib)==1)then!inlet
                            neighbxm(ipx,i) = 0
                        elseif(bound_type(ib)==2)then!outflow
                            neighbxm(ipx,i) = npx
                        elseif(bound_type(ib)==3)then!cyclic
                            neighbxm(ipx,i) = pt_index-npx+1
                        elseif(bound_type(ib)==4)then!sym
                            neighbxm(ipx,i) = 2*npx-pt_index
                        else
                            write(*,*)'Wrong boundary type'
                            stop
                        endif
                    else
                        neighbxm(ipx,i)=pt_index
                    endif
                enddo
            enddo
            !grid
            do ipx=1,npx
                do i=1,5
                    pt_index=ipx-3+i
                    if(pt_index<1)then
                        if(bound_type(ib)==1)then!inlet
                            neighbx(ipx,i) = 0
                        elseif(bound_type(ib)==2)then!outflow
                            neighbx(ipx,i) = 1
                        elseif(bound_type(ib)==3)then!cyclic
                            neighbx(ipx,i) = pt_index+npx-1
                        elseif(bound_type(ib)==4)then!sym
                            neighbx(ipx,i) = 2-pt_index
                        else
                            write(*,*)'Wrong boundary type'
                            stop
                        endif
                    elseif(pt_index>npx)then
                        if(bound_type(ib)==1)then!inlet
                            neighbx(ipx,i) = 0
                        elseif(bound_type(ib)==2)then!outflow
                            neighbx(ipx,i) = npx
                        elseif(bound_type(ib)==3)then!cyclic
                            neighbx(ipx,i) = pt_index-npx+1
                        elseif(bound_type(ib)==4)then!sym
                            neighbx(ipx,i) = 2*npx-pt_index
                        else
                            write(*,*)'Wrong boundary type'
                            stop
                        endif
                    else
                        neighbx(ipx,i)=pt_index
                    endif
                enddo
            enddo
        elseif(ib==3.or.ib==4)then!y-direction
            !face
            do ipy=1,npy+1
                !f+
                do i=1,5
                    pt_index=ipy-4+i
                    if(pt_index.lt.1)then
                        if(bound_type(ib)==1)then!inlet
                            neighbyp(ipy,i) = 0
                        elseif(bound_type(ib)==2)then!outflow
                            neighbyp(ipy,i) = 1
                        elseif(bound_type(ib)==3)then!cyclic
                            neighbyp(ipy,i) = npy+pt_index-1
                        elseif(bound_type(ib)==4)then!sym
                            neighbyp(ipy,i) = 2-pt_index
                        else
                            write(*,*)'Wrong boundary type'
                            stop
                        endif
                    elseif(pt_index.gt.npy)then
                        if(bound_type(ib)==1)then!inlet
                            neighbyp(ipy,i) = 0
                        elseif(bound_type(ib)==2)then!outflow
                            neighbyp(ipy,i) = npy
                        elseif(bound_type(ib)==3)then!cyclic
                            neighbyp(ipy,i) = pt_index-npy+1
                        elseif(bound_type(ib)==4)then!sym
                            neighbyp(ipy,i) = 2*npy-pt_index
                        else
                            write(*,*)'Wrong boundary type'
                            stop
                        endif
                    else
                        neighbyp(ipy,i)=pt_index
                    endif
                enddo
                !f-
                do i=1,5
                    pt_index=ipy+3-i
                    if(pt_index.lt.1)then
                        if(bound_type(ib)==1)then!inlet
                            neighbym(ipy,i) = 0
                        elseif(bound_type(ib)==2)then!outflow
                            neighbym(ipy,i) = 1
                        elseif(bound_type(ib)==3)then!cyclic
                            neighbym(ipy,i) = npy+pt_index-1
                        elseif(bound_type(ib)==4)then!sym
                            neighbym(ipy,i) = 2-pt_index
                        else
                            write(*,*)'Wrong boundary type'
                            stop
                        endif
                    elseif(pt_index.gt.npy)then
                        if(bound_type(ib)==1)then!inlet
                            neighbym(ipy,i) = 0
                        elseif(bound_type(ib)==2)then!outflow
                            neighbym(ipy,i) = npy
                        elseif(bound_type(ib)==3)then!cyclic
                            neighbym(ipy,i) = pt_index-npy+1
                        elseif(bound_type(ib)==4)then!sym
                            neighbym(ipy,i) = 2*npy-pt_index
                        else
                            write(*,*)'Wrong boundary type'
                            stop
                        endif
                    else
                        neighbym(ipy,i)=pt_index
                    endif
                enddo
            enddo
            !grid
            do ipy=1,npy
                do i=1,5
                    pt_index=ipy-3+i
                    if(pt_index<1)then
                        if(bound_type(ib)==1)then!inlet
                            neighby(ipy,i) = 0
                        elseif(bound_type(ib)==2)then!outflow
                            neighby(ipy,i) = 1
                        elseif(bound_type(ib)==3)then!cyclic
                            neighby(ipy,i) = pt_index+npy-1
                        elseif(bound_type(ib)==4)then!sym
                            neighby(ipy,i) = 2-pt_index
                        else
                            write(*,*)'Wrong boundary type'
                            stop
                        endif
                    elseif(pt_index>npy)then
                        if(bound_type(ib)==1)then!inlet
                            neighby(ipy,i) = 0
                        elseif(bound_type(ib)==2)then!outflow
                            neighby(ipy,i) = npy
                        elseif(bound_type(ib)==3)then!cyclic
                            neighby(ipy,i) = pt_index-npy+1
                        elseif(bound_type(ib)==4)then!sym
                            neighby(ipy,i) = 2*npy-pt_index
                        else
                            write(*,*)'Wrong boundary type'
                            stop
                        endif
                    else
                        neighby(ipy,i)=pt_index
                    endif
                enddo
            enddo
        endif
    enddo
    !test
    !do ipx=1,npx+1
    !    !f+
    !    do i=1,5
    !        pt_index = ipx-4+i
    !        if(pt_index.lt.1)then
    !            !tmp=npx+pt_index-1!cyc
    !            tmp = 2-pt_index!sym
    !        elseif(pt_index.gt.npx)then
    !            !tmp=pt_index-npx+1!cyc
    !            tmp = 2*npx-pt_index
    !        else
    !            tmp=pt_index
    !        endif
    !        if(tmp.ne.neighbxp(ipx,i))then
    !            write(*,*)ipx,i,tmp,neighbxp(ipx,i)
    !        endif
    !    enddo
    !    !f-
    !    do i=1,5
    !        pt_index = ipx+3-i
    !        if(pt_index.lt.1)then
    !            !tmp=npx+pt_index-1!cyc
    !            tmp = 2-pt_index!sym
    !        elseif(pt_index.gt.npx)then
    !            !tmp=pt_index-npx+1!cyc
    !            tmp = 2*npx-pt_index!sym
    !        else
    !            tmp=pt_index
    !        endif
    !        if(tmp.ne.neighbxm(ipx,i))then
    !            write(*,*)ipx,i,tmp,neighbxm(ipx,i)
    !        endif
    !    enddo
    !enddo
    !do ipy=1,npy+1
    !    !f+
    !    do i=1,5
    !        pt_index = ipy-4+i
    !        if(pt_index.lt.1)then
    !            !tmp=npy+pt_index-1!cyc
    !            tmp = 2-pt_index!sym
    !        elseif(pt_index.gt.npy)then
    !            !tmp=pt_index-npy+1!cyc
    !            tmp = 2*npy-pt_index!sym
    !        else
    !            tmp=pt_index
    !        endif
    !        if(tmp.ne.neighbyp(ipy,i))then
    !            write(*,*)ipy,i,tmp,neighbyp(ipy,i)
    !        endif
    !    enddo
    !    !f-
    !    do i=1,5
    !        pt_index = ipy+3-i
    !        if(pt_index.lt.1)then
    !            !tmp=npy+pt_index-1!cyc
    !            tmp = 2-pt_index!sym
    !        elseif(pt_index.gt.npy)then
    !            !tmp=pt_index-npy+1!cyc
    !            tmp = 2*npy-pt_index!sym
    !        else
    !            tmp=pt_index
    !        endif
    !        if(tmp.ne.neighbym(ipy,i))then
    !            write(*,*)ipy,i,tmp,neighbym(ipy,i)
    !        endif
    !    enddo
    !enddo
    !do ipx=1,npx
    !    do i=1,5
    !        pt_index = ipx-3+i
    !        if(pt_index < 1)then
    !            !tmp = pt_index+npx-1!cyc
    !            tmp = 2-pt_index!sym
    !        elseif(pt_index > npx)then
    !            !tmp = pt_index-npx+1!cyc
    !            tmp=2*npx-pt_index
    !        else
    !            tmp = pt_index
    !        endif
    !        if(tmp.ne.neighbx(ipx,i))then
    !            write(*,*)ipx,i,tmp,neighbx(ipx,i)
    !        endif
    !    enddo
    !enddo
    !do ipy=1,npy
    !    do i=1,5
    !        pt_index = ipy-3+i
    !        if(pt_index < 1)then
    !            !tmp = pt_index+npy-1!cyc
    !            tmp = 2-pt_index!sym
    !        elseif(pt_index > npy)then
    !            !tmp = pt_index-npy+1!cyc
    !            tmp = 2*npy-pt_index!sym
    !        else
    !            tmp = pt_index
    !        endif
    !        if(tmp.ne.neighby(ipy,i))then
    !            write(*,*)ipy,i,tmp,neighby(ipy,i)
    !        endif
    !    enddo
    !enddo
    !write(*,*)'done'
    !stop
    
end subroutine boundary_condition_cpu

subroutine inlet_condition(bound)
    use setup
    implicit none

    integer, intent(in):: bound

    
end subroutine inlet_condition

subroutine outflow_condition(bound)
    use setup
    implicit none

    integer, intent(in):: bound

    
end subroutine outflow_condition

subroutine cyclic_condition(bound)
    use setup
    implicit none

    integer, intent(in):: bound
    integer::iface,i

    if(bound==1)then!xmin

    elseif(bound==2)then!xmax

    elseif(bound==3)then!ymin

    elseif(bound==4)then!ymax

    endif
    
end subroutine cyclic_condition

subroutine sym_condition(bound)
    use setup
    implicit none

    integer, intent(in):: bound

    
end subroutine sym_condition


subroutine write_solu_tecplot
    use setup
    implicit none

    character*6 :: decimal
    character*64 :: filename, full_path
    integer :: ipx,ipy,icx,icy,ipc
    double precision :: rho,u,v,w,pr,Bx,By,Bz

    write(decimal,'(f6.3)') ctime-int(ctime)
    write(filename,'(a,i7.7,a4,a)')'TEC',int(ctime),decimal(3:6),'.dat'
    write(full_path,'(A,"/",A)') trim(folder_path), trim(filename)
    open(11,file=full_path)
    write(11,*) 'TITLE = "MHD"'
    write(11,*) 'VARIABLES = "X" "Y" "RHO" "U" "V" "P" "Bx" "By" "Bz"'
    write(11,12) 'ZONE NODES=',npx*npy,',ELEMENTS=',(npx-1)*(npy-1),&
            ',DATAPACKING=POINT,ZONETYPE=FEQUADRILATERAL'
    12  FORMAT(A11,I10,A10,I10,A43)

    do ipy = 1,npy
        do ipx = 1,npx

            rho = ub(ipx,ipy,1)
            u = ub(ipx,ipy,2)/rho
            v = ub(ipx,ipy,3)/rho
            w = ub(ipx,ipy,4)/rho
            Bx = ub(ipx,ipy,6)
            By = ub(ipx,ipy,7)
            Bz = ub(ipx,ipy,8)
            pr = (gam-1.d0)*(ub(ipx,ipy,5)-0.5d0*(u**2+v**2+w**2)*rho&
                    -0.5d0*(Bx**2+By**2+Bz**2))


            write(11,*) xmin+dble(ipx-1)*hx,ymin+dble(ipy-1)*hy,rho,&
            u,v,pr, Bx, By, Bz

        end do
    end do

    do icy = 1,npy-1
        do icx = 1,npx-1
            ! global index for the left bottom corner point
            ! for a square element
            ipc = (icy-1)*npx+icx
            write(11,*) ipc,ipc+1,ipc+1+npx,ipc+npx
        end do
    end do
    close(11)

end subroutine write_solu_tecplot

subroutine write_data
    use setup
    implicit none

    integer :: io_status
    character*6 :: decimal
    character*64 :: filename, full_path

    write(decimal,'(f6.3)') ctime-int(ctime)
    write(filename,'(a,i7.7,a4,a)')'restart.',int(ctime),decimal(3:6),'.dat'
    write(full_path,'(A,"/",A)') trim(folder_path), trim(filename)

    open(unit=21,file=full_path,form="unformatted",&
            status="replace",action="write",iostat=io_status)
    write(21) ctime,iter,ub
    close(21)

end subroutine write_data

subroutine read_data
    use setup
    implicit none

    integer :: io_status
    character*64::full_path
    write(full_path,'(A,"/",A)') trim(folder_path), trim(namerestart)
    open(unit=22,file=full_path,form="unformatted",&
            status="old",action="read",iostat=io_status)
    read(22) ctime,iter,ub
    close(22)

end subroutine read_data