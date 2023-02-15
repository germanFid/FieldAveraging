module FieldAveragingModule
	implicit none
	
	contains

	function AverageThisField(OurField) Result(result)
		implicit none
		
		real*16, dimension(:,:) :: OurField
		integer :: k=1
		
		integer :: n
		integer :: m
		integer :: i, j
		integer :: i_start, j_start
		integer :: i_end, j_end
		integer :: ii, jj
		real*16 :: SumOfElements
		real*16 :: NumberOfElements
		real*16 :: AverageValue
		real*16, dimension(:,:), allocatable :: result
		
		n=size(OurField(:,1))
		m=size(OurField(1,:))
		
		allocate(result(n,m))
		
		do i=1, n
			do j=1, m
			
				i_start = i-k
				if(i_start<1) i_start=1
				j_start = j-k
				if(j_start<1) j_start=1
				i_end = i+k
				if(i_end>n) i_end=n
				j_end = j+k
				if(j_end>m) j_end=m
				
				NumberOfElements=(i_end-i_start+1)*(j_end-j_start+1)
				SumOfElements=0
				do ii=i_start, i_end
					do jj=j_start, j_end
						SumOfElements=SumOfElements+OurField(ii,jj)
					end do
				end do
				AverageValue=SumOfElements/NumberOfElements
				
				do ii=i_start, i_end
					do jj=j_start, j_end
						OurField(ii,jj)=AverageValue
					end do
				end do
			end do
		end do
		
		!write(*,*) 
		!write(*,*) OurField
		
		result=OurField
	end function
	
end module

program FieldAveraging
	use FieldAveragingModule
	implicit none
	
	integer :: n, m 
	integer :: i, j, k, l, p, flag
	real*16, dimension(:,:), allocatable :: OurField
	real*16, dimension(:,:), allocatable :: Temp
	character(len=4) :: str_num
	real*16 :: Eps
	real*16 :: t_start, t_finish
	
	Eps = 0.1
	
	open(21, file = 'iter_size.txt')
	
	do k=5, 290, 5
		
		if (k<10) then
			write(str_num, '(i1)') k
		elseif (k<100) then
			write(str_num, '(i2)') k
		elseif (k<1000) then
			write(str_num, '(i3)') k
		else
			write(str_num, '(i4)') k
		end if
		
		open(20, file= 'Matrices/'//trim(str_num)//'.txt')
		
		n = k
		m = 5
		
		allocate(OurField(n,m))
		allocate(Temp(n,m))
		
		do i=1,n
			read(20, *) (OurField(i,j), j=1,m)
		end do
		
		close(20)
		
		i = 0
		flag = 0
		
		call cpu_time(t_start)
		
		do while (flag == 0)
		
			i = i+1
			Temp=AverageThisField(OurField)
			
			if (abs(temp(n,m)-1)<Eps) then
				flag = 1
			end if
			
		end do
		
		call cpu_time(t_finish)
		
		write(21,*) i, k, t_finish-t_start
		
		deallocate(OurField)
		deallocate(Temp)
		
	end do
	
	k = 100
	Eps = 1
	open(20, file= 'Matrices/100.txt')
		
	n = k
	m = 5
		
	allocate(OurField(n,m))
	allocate(Temp(n,m))
		
	do i=1,n
		read(20, *) (OurField(i,j), j=1,m)
	end do
		
	close(20)
	
	open(22, file = 'Eps_iter.txt')
	
	do p = 1, 10
	
		Eps = Eps/10.0
		
		i = 0
		flag = 0
		
		call cpu_time(t_start)
		
		do while (flag == 0)
		
			i = i+1
			Temp=AverageThisField(OurField)
			
			if (abs(temp(n,m)-1)<Eps) then
				flag = 1
			end if
			
		end do
		
		call cpu_time(t_finish)
		
		write(22,*) eps, p, t_finish-t_start

	end do
	
	deallocate(OurField)
	deallocate(Temp)
end program