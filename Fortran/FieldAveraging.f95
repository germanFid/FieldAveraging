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
		
		write(*,*) 
		write(*,*) OurField
		
		result=OurField
	end function
	
end module

program FieldAveraging
	use FieldAveragingModule
	implicit none
	
	integer :: n, m 
	integer :: i, j
	real*16, dimension(:,:), allocatable :: OurField
	real*16, dimension(:,:), allocatable :: Temp
	
	open(20, file="InputFiles\InputData.plt")
	
	read(20, *) n, m
	
	allocate(OurField(n,m))
	allocate(Temp(n,m))
	
	do i=1,n
		read(20, *) (OurField(i,j), j=1,m)
	end do
	
	close(20)
	
	do i=1,5
		Temp=AverageThisField(OurField)
	end do
	
end program