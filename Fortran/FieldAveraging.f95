module FieldAveragingModule
	implicit none
	
	contains

	function SquareAveraging(OurField,i,j,k) result(result)
		implicit none
		
		real*16, dimension(:,:) :: OurField
		
		integer :: k
		integer :: n, m
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
		
		result=OurField
		
	end function

	function AverageThisField(OurField) Result(result)
		implicit none
		
		real*16, dimension(:,:) :: OurField
		integer :: k=1
		
		integer :: n, m
		integer :: i, j
		real*16, dimension(:,:), allocatable :: result
		
		n=size(OurField(:,1))
		m=size(OurField(1,:))
		
		allocate(result(n,m))
		
		write(*,*) "AverageThisField"
		
		do i=1, n
			do j=1, m
				
				write(*,*) "ElementNumber=", i, ",",j
				
				OurField=SquareAveraging(OurField,i,j,k)
				
			end do
		end do
		
		result=OurField
	end function
	
	function DiagonalAverageThisField(OurField) Result(result)
		implicit none
		
		real*16, dimension(:,:) :: OurField
		integer :: k=1
		
		integer :: n, m
		integer :: i, j
		integer :: i_begin, j_begin
		real*16, dimension(:,:), allocatable :: result
		
		n=size(OurField(:,1))
		m=size(OurField(1,:))
		
		allocate(result(n,m))
		
		write(*,*) "DiagonalAverageThisField"
		
		do i_begin=1, n
			i=i_begin
			j=1
			do while((i>=1).and.(j<=m))
			
				write(*,*) "ElementNumber=", i, ",",j
				
				OurField=SquareAveraging(OurField,i,j,k)
				
				if(i>=1) i=i-1
				if(j<=m) j=j+1
				
			end do
		end do
		
		do j_begin=2, m
			j=j_begin
			i=n
			do while((i>=1).and.(j<=m))
			
				write(*,*) "ElementNumber=", i, ",",j
				
				OurField=SquareAveraging(OurField,i,j,k)
				
				if(i>=1) i=i-1
				if(j<=m) j=j+1
				
			end do
		end do
		
		result=OurField
	end function
	
end module

program FieldAveraging
	use FieldAveragingModule
	implicit none
	
	integer :: n, m 
	integer :: i, j
	real*16, dimension(:,:), allocatable :: InputField
	real*16, dimension(:,:), allocatable :: TempArray
	
	open(20, file="InputFiles\InputData.plt")
	read(20, *) n, m
	
	allocate(InputField(n,m))
	allocate(TempArray(n,m))
	
	do i=1,n
		read(20, *) (InputField(i,j), j=1,m)
	end do
	
	close(20)
	write(*,*) InputField
	
	do i=1,1
		write(*,*) "GlobalIterNumber=", i
		TempArray=AverageThisField(InputField)
	end do
	write(*,*) TempArray
	
	open(20, file="InputFiles\InputData.plt")
	read(20, *) n, m
	
	do i=1,n
		read(20, *) (InputField(i,j), j=1,m)
	end do
	
	close(20)
	write(*,*) InputField
	
	do i=1,1
		write(*,*) "GlobalIterNumber=", i
		TempArray=DiagonalAverageThisField(InputField)
	end do
	write(*,*) TempArray
	 
end program