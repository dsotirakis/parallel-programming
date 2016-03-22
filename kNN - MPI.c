#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <mpi.h>


int NQ;         
int NC;        
float dims;          
int P;


struct dimens{
  double x_dim;
  double y_dim;
  double z_dim;
  int id;
};

struct boxes{
  double x_dim;
  double y_dim;
  double z_dim;
  int num_of_q, num_of_c;
  int id;
  int process_rank;
};

int cmpfuncx (const void * p1, const void * p2)
{
    const struct dimens *elem1 = p1;    
    const struct dimens *elem2 = p2;

    if ( elem1->x_dim < elem2->x_dim)
      return -1;
    else if (elem1->x_dim > elem2->x_dim)
      return 1;
    else
      return 0;
}

int cmpfuncy (const void * p1, const void * p2)
{
    const struct dimens *elem1 = p1;    
    const struct dimens *elem2 = p2;

    if ( elem1->y_dim < elem2->y_dim)
      return -1;
    else if (elem1->y_dim > elem2->y_dim)
      return 1;
    else
      return 0;
}

int cmpfuncz (const void * p1, const void * p2)
{
    const struct dimens *elem1 = p1;    
    const struct dimens *elem2 = p2;

    if ( elem1->z_dim < elem2->z_dim)
      return -1;
    else if (elem1->z_dim > elem2->z_dim)
      return 1;
    else
      return 0;
}

double Log2( double n )  
{  
    // log(n)/log(2) is log2.  
    return log( n ) / log( 2 );  
}


int main(int argc, char **argv) {
 
	if (argc != 5) {
	printf("Usage: %s q\n  where n=2^q is problem size (power of two)\n", 
	   argv[0]);
	exit(1);
	}

    //Initializing inputs//
	NQ = 1<<atoi(argv[1]);
	struct dimens *qCoords = (struct dimens *)malloc(NQ * sizeof(struct dimens));
	NC = 1<<atoi(argv[2]);
	struct dimens *cCoords = (struct dimens *)malloc(NC * sizeof(struct dimens));
	dims = 1<<atoi(argv[3]);
	P  = 1<<atoi(argv[4]);


	int xa,n,m,k;
	int temp2=0;
	double res=0;
	res=dims/P;
	res=res/3;
	res=Log2(res);
	n=res;
	m=res;
	k=res;
	int ela=0;
	while (n%2==1 || m%2==1 || k%2==1){
		if (ela%3==0){
			k++;
		}
		else if (ela%3==1){
			m++;
		}
		else{
			n++;
		}
		ela++;

	}
	res=P/3;
	if(res>1){
		res=Log2(res);
	}
	int na=res;
	int ma=res;
	int ka=res;
	ela=0;
	while ((ka*ma*na)!=P){
		if (ela%3==0){
			ka++;
		}
		else if (ela%3==1){
			ma++;
		}
		else{
			na++;
		}
		ela++;
	}
	
	struct dimens *qItemsToSend = (struct dimens *)malloc(NQ * sizeof(struct dimens));
	struct dimens *cItemsToSend = (struct dimens *)malloc(NC * sizeof(struct dimens));

	struct timeval t1,t2, startwtime, endwtime;


	int i,j,l;

	double x_dim, y_dim, z_dim;
	int id;

	int rank, size;
	
 	
  	MPI_Init (&argc, &argv);      				  /* starts MPI */
  	MPI_Comm_rank (MPI_COMM_WORLD, &rank);        /* get current process id */
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    //Creating custom Datatype//

	MPI_Datatype structtype;
	int          blockcounts[2];
	MPI_Aint     offsets[2];
	MPI_Datatype oldtypes[2];

	/* Setup description of the 4 MPI_FLOAT fields x, y, z, velocity */
	offsets[0] = 0;
	oldtypes[0] = MPI_DOUBLE;
	blockcounts[0] = 3;

	/* Setup description of the 2 MPI_INT fields n, type */
	/* Need to first figure offset by getting size of MPI_FLOAT */
	offsets[1] = 3 * 8;
	oldtypes[1] = MPI_INT;
	blockcounts[1] = 1;

	/* Now define structured type and commit it */
	MPI_Type_struct(2, blockcounts, offsets, oldtypes, &structtype);
	MPI_Type_commit(&structtype);

	double rank_n,rank_m,rank_k;
	int rank_mf;

	//Defining variables to use in each process so as to
	//define the start point of each rank
	rank_n= rank%na;
	rank_m=rank/ma;
	rank_k=rank/(ma*na);
	rank_mf=-1;
	rank_k=rank_k/(double)ka;
	if (rank>=ma*na){
		rank_m=rank-(rank/(ma*na))*(ma*na);
		rank_mf= rank_m/ma;
		rank_m=(double)rank_mf/(double)ma;
	}
	else{
		rank_m=rank_m/ma;
	}
	if ((size==2 || size==4) && (rank==1 || rank==3)){
		rank_m= 1/(double)ma;
	}

	//Initializing Grid's dimensions//
	gettimeofday (&startwtime, NULL);
	struct boxes *gridBoxesPerProcess = (struct boxes *)malloc(10000 * sizeof(struct boxes));
	gridBoxesPerProcess[0].process_rank=rank;
	int counter=0;
	double point;

	if (rank==0){
		for (i=0; i<(k/ka); i++){
			for (j=0; j<(m/ma); j++){
				for (l=0; l<(n/na); l++){
					gridBoxesPerProcess[counter].x_dim=(l/((double)n));
					gridBoxesPerProcess[counter].y_dim=(j/((double)m));
					gridBoxesPerProcess[counter].z_dim=(i/((double)k));
					gridBoxesPerProcess[counter].id=counter;
					counter++;
				}
			}
			if (i==1){
				point=gridBoxesPerProcess[counter-1].z_dim;
			}
		}
	}
	else{
		for (i=0; i<k/ka; i++){
			for (j=0; j<m/ma; j++){
				for (l=0; l<n/na; l++){
					gridBoxesPerProcess[counter].x_dim=(l/((double)n)) + (rank_n/na);
					gridBoxesPerProcess[counter].y_dim=(j/((double)m)) + (rank_m);
					gridBoxesPerProcess[counter].z_dim=(i/((double)k)) + (rank_k);
					gridBoxesPerProcess[counter].id=counter;
					counter++;
				}
			}
			if (i==1){
				point=gridBoxesPerProcess[counter-1].z_dim;
			}
		}

	}

	double x_dist=gridBoxesPerProcess[1].x_dim-gridBoxesPerProcess[0].x_dim;
	double y_dist=gridBoxesPerProcess[m].y_dim-gridBoxesPerProcess[m-1].y_dim;
	double z_dist=point-gridBoxesPerProcess[0].z_dim;
	int num_of_cs=0, num_of_qs=0, num_of_cs_to_send=0, num_of_qs_to_send=0;


	gettimeofday(&t1, NULL);
	srand((rank+1)*t1.tv_usec * t1.tv_sec);

	////////////////////////////////////////Generating Numbers/////////////////////////////////////
	//First generate numbers. Then keep the numbers that belong to each rank's borders, and if not
	//put them on the array for sending
		for (i = 0; i < NQ/P; i++){
		x_dim = ((double)rand()/(double)RAND_MAX);
		y_dim = ((double)rand()/(double)RAND_MAX);
		z_dim = ((double)rand()/(double)RAND_MAX);
		id = i;
		if ((x_dim>=gridBoxesPerProcess[0].x_dim && x_dim <=gridBoxesPerProcess[counter-1].x_dim + x_dist)
			&& (y_dim>=gridBoxesPerProcess[0].y_dim && y_dim <=gridBoxesPerProcess[counter-1].y_dim + y_dist)
			&& (z_dim>=gridBoxesPerProcess[0].z_dim && z_dim <=gridBoxesPerProcess[counter-1].z_dim + z_dist)
			&& (x_dim!=0 && y_dim!=0 && z_dim!=0))
		{
			qCoords[num_of_qs].x_dim=x_dim;
			qCoords[num_of_qs].y_dim=y_dim;
			qCoords[num_of_qs].z_dim=z_dim;
			qCoords[num_of_qs].id=id;
			num_of_qs++;
		}
		else
		{
			qItemsToSend[num_of_qs_to_send].x_dim=x_dim;
			qItemsToSend[num_of_qs_to_send].y_dim=y_dim;
			qItemsToSend[num_of_qs_to_send].z_dim=z_dim;
			qItemsToSend[num_of_qs_to_send].id=id;
			num_of_qs_to_send++;
		}
	}
	if (size==2){
		gridBoxesPerProcess[0].y_dim=0;
	}
	gettimeofday(&t2, NULL);
	srand((rank+1)*t2.tv_usec * t2.tv_sec);

	for (i = 0; i < NC/P; i++){
		x_dim = ((double)rand()/(double)RAND_MAX);
		y_dim = ((double)rand()/(double)RAND_MAX);
		z_dim = ((double)rand()/(double)RAND_MAX);
		id = i;
		if ((x_dim>=gridBoxesPerProcess[0].x_dim && x_dim <=gridBoxesPerProcess[counter-1].x_dim + x_dist)
			&& (y_dim>=gridBoxesPerProcess[0].y_dim && y_dim <=gridBoxesPerProcess[counter-1].y_dim + y_dist)
			&& (z_dim>=gridBoxesPerProcess[0].z_dim && z_dim <=gridBoxesPerProcess[counter-1].z_dim + z_dist))
		{
			cCoords[num_of_cs].x_dim=x_dim;	
			cCoords[num_of_cs].y_dim=y_dim;
			cCoords[num_of_cs].z_dim=z_dim;
			cCoords[num_of_cs].id=id;
			num_of_cs++;
		}
		else
		{
			cItemsToSend[num_of_cs_to_send].x_dim=x_dim;
			cItemsToSend[num_of_cs_to_send].y_dim=y_dim;
			cItemsToSend[num_of_cs_to_send].z_dim=z_dim;
			cItemsToSend[num_of_cs_to_send].id=id;
			num_of_cs_to_send++;
		}
	}

	int Qsentcnt[1]={num_of_qs_to_send-1};
	int Csentcnt[1]={num_of_cs_to_send-1};
	int totalQ[1]={0};
	int totalC[1]={0};

	//Use of Allreduce: Every process will know the size of ALL elements that are goint to be sent.//

	MPI_Allreduce(Qsentcnt, totalQ, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(Csentcnt, totalC, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	int displs[size], qarray[size], carray[size];
	qarray[rank]=Qsentcnt[0];
	carray[rank]=Csentcnt[0];

	struct dimens *qrecv = (struct dimens *)malloc(totalQ[0] * sizeof(struct dimens));
	struct dimens *crecv = (struct dimens *)malloc(totalC[0] * sizeof(struct dimens));
	int qr[size];
	int cr[size];

	//Use of Allgather: Every i-th element of arrays qarray and carray have the number of elements
	//that i rank is going to send.

	MPI_Allgather(&qarray[rank], 1, MPI_INT, &qr[0], 1, MPI_INT, MPI_COMM_WORLD);
	MPI_Allgather(&carray[rank], 1, MPI_INT, &cr[0], 1, MPI_INT, MPI_COMM_WORLD);

	//displs: array of displacement. According to MPI_Allgatherv syntax, we need to pass the correct place that
	//rank i is going to begin passing arguments. For example if displs[1] is 100, MPI_Allgatherv, sends
	//the qItemsToSend, of number Qsentcnt, of Datatype structtype to the array qrecv, which number is totalQ.
	//The items qrecv receives from i-th rank, for each rank, are of number qr[i], and are placed after displs[i]
	//items. So process 1 is going to place its elements after the first 100 elements of process 0.
	displs[0]=0;
	int temp=0;
	for (i=1; i<size; i++){
		temp+=qr[i-1];
		displs[i]=temp;
	}

	MPI_Allgatherv(&qItemsToSend[0], Qsentcnt[0], structtype, &qrecv[0], qr, displs, structtype, MPI_COMM_WORLD);
	MPI_Allgatherv(&cItemsToSend[0], Csentcnt[0], structtype, &crecv[0], cr, displs, structtype, MPI_COMM_WORLD);

	//At this point, every process has all elements from all processes. Now we need to check which elements belongs
	//to this process for each Q and C.

	//At this point, needs to be refered that MPI_Barrier should do the work, but I found more functional to 
	//use the methods above.

		temp=1;
	struct dimens *qFinal = (struct dimens *)malloc(sizeof(struct dimens));
	for (i=0; i<totalQ[0]; i++){
		if ((qrecv[i].x_dim>=gridBoxesPerProcess[0].x_dim && qrecv[i].x_dim <=gridBoxesPerProcess[counter-1].x_dim + x_dist)
			&& (qrecv[i].y_dim>=gridBoxesPerProcess[0].y_dim && qrecv[i].y_dim <=gridBoxesPerProcess[counter-1].y_dim + y_dist)
			&& (qrecv[i].z_dim>=gridBoxesPerProcess[0].z_dim && qrecv[i].z_dim <=gridBoxesPerProcess[counter-1].z_dim + z_dist)
			&& (qrecv[i].x_dim!=0 && qrecv[i].y_dim!=0 && qrecv[i].z_dim!=0))
		{
			qFinal[temp-1]=qrecv[i];
			temp++;
			if (temp==totalQ[0]){
				break;
			}
			qFinal=(struct dimens *)realloc(qFinal, temp*sizeof(struct dimens));

		}
	}


	int temp1=temp;
	qFinal=(struct dimens *)realloc(qFinal, (temp+num_of_qs-2)*sizeof(struct dimens));


	int count=0;
	for (i=(temp-2); i<(temp+num_of_qs-2); i++){
		qFinal[i]=qCoords[count];
		count++;

	}
		temp=1;

	struct dimens *cFinal = (struct dimens *)malloc(sizeof(struct dimens));
	for (i=0; i<totalC[0]; i++){
		if ((crecv[i].x_dim>=gridBoxesPerProcess[0].x_dim && crecv[i].x_dim <=gridBoxesPerProcess[counter-1].x_dim + x_dist)
			&& (crecv[i].y_dim>=gridBoxesPerProcess[0].y_dim && crecv[i].y_dim <=gridBoxesPerProcess[counter-1].y_dim + y_dist)
			&& (crecv[i].z_dim>=gridBoxesPerProcess[0].z_dim && crecv[i].z_dim <=gridBoxesPerProcess[counter-1].z_dim + z_dist)
			&& (crecv[i].x_dim!=0 && crecv[i].y_dim!=0 && crecv[i].z_dim!=0))
		{
			cFinal[temp-1]=crecv[i];
			temp++;
			cFinal=(struct dimens *)realloc(cFinal, temp*sizeof(struct dimens));
		}
	}
	cFinal=(struct dimens *)realloc(cFinal, (temp+num_of_cs-3)*sizeof(struct dimens));
	count=0;
	for (i=(temp-2); i<(temp+num_of_cs-2); i++){
		cFinal[i]=cCoords[count];
		count++;
	}


	qsort(cFinal,(temp+num_of_cs-1),sizeof(struct dimens),cmpfuncx);
	qsort(qFinal,(temp1+num_of_qs-1),sizeof(struct dimens),cmpfuncx);

	//Gathering times, in order to find the quickest and the slowest process to complete.

	gettimeofday (&endwtime, NULL);
	double seq_time1[size];
 	seq_time1[rank] = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);


 	double time1[size];
 	MPI_Allgather(&seq_time1[rank],1,MPI_DOUBLE,&time1[0],1,MPI_DOUBLE,MPI_COMM_WORLD);

 	double max_time=-1,min_time=10;
 	int max_point,min_point;
 	for (i=0; i<size; i++){
 		if (time1[i]<=min_time){
 			min_time=time1[i];
 			min_point=i;
 		}
 		if (time1[i]>=max_time){
 			max_time=time1[i];
 			max_point=i;
 		}
 	}

  	gettimeofday (&startwtime, NULL);
  	double min_dist_x=gridBoxesPerProcess[0].x_dim, min_dist_y=gridBoxesPerProcess[0].y_dim, min_dist_z=gridBoxesPerProcess[0].z_dim;
	double max_dist_x=gridBoxesPerProcess[n-1].x_dim+x_dist-min_dist_x;
	double max_dist_y=gridBoxesPerProcess[(n*m)-1].y_dim+y_dist-min_dist_y;
	double max_dist_z=gridBoxesPerProcess[counter-1].z_dim+z_dist-min_dist_z;
	if (size==2){
		min_dist_y=0;
	}


	id=-1;
	int flag=0;
	int flags[12]={0,0,0,0,0,0,0,0,0,0,0,0};
	int waitFlags[6]={0,0,0,0,0,0};
	MPI_Request *reqs;
	MPI_Status *stats;
	int times=0;
	int nums[6];
	int sum_nums[size*6];
	for (i=0; i<6; i++){
		displs[i]=6;
	}
	reqs = (MPI_Request *) malloc(24*sizeof(MPI_Request));
	stats = (MPI_Status *) malloc(20*sizeof(MPI_Status));

	//Sending methods. I use MPI_Isend, if process has other process near it.
	//If that is so, it needs to send and receive elements because there may be
	//Qs that needs to communicate to find their closest C.

	if (gridBoxesPerProcess[0].x_dim-max_dist_x>=0 && flags[0]!=1){
		flags[0]=1;
		struct dimens *cToComm = (struct dimens *)malloc(sizeof(struct dimens));
		i=1;
		while (cFinal[i-1].x_dim<=gridBoxesPerProcess[0].x_dim+x_dist){
			cToComm[i-1]=cFinal[i];
			i++;
			cToComm=(struct dimens *)realloc(cToComm, i*sizeof(struct dimens));
		}
		nums[0]=i-1;
		MPI_Isend(&nums[0],1,MPI_INT,rank-1,1+(10*rank),MPI_COMM_WORLD,&reqs[6]);
		MPI_Isend(&cToComm[0],i-1,structtype,rank-1,1+(10*rank),MPI_COMM_WORLD,&reqs[0]);
		printf("Rank=%d. Send to rank=%d\n",rank, rank-1);


	}
	if (gridBoxesPerProcess[0].x_dim+x_dist+max_dist_x<=1 && flags[1]!=1){
		flags[1]=1;
		struct dimens *cToComm = (struct dimens *)malloc(sizeof(struct dimens));
		i=0;
		int point=-1;
		for(;;){
			if (cFinal[i].x_dim>=gridBoxesPerProcess[n-1].x_dim){
				point=i;
				break;
			}
			i++;
		}
		i=1;
		while (cFinal[point+i-1].x_dim<=gridBoxesPerProcess[n-1].x_dim+x_dist && (point+i-1)<=temp+num_of_cs-2){
			cToComm[i-1]=cFinal[point+i-1];
			i++;
			cToComm=(struct dimens *)realloc(cToComm, i*sizeof(struct dimens));
		}
		nums[1]=i-1;
		MPI_Isend(&nums[1],1,MPI_INT,rank+1,2+(10*rank),MPI_COMM_WORLD,&reqs[7]);
		MPI_Isend(&cToComm[0],i-1,structtype,rank+1,2+(10*rank),MPI_COMM_WORLD,&reqs[1]);
		printf("Rank=%d. Send to rank=%d\n",rank, rank+1);

	}
	if (gridBoxesPerProcess[0].y_dim-max_dist_y>=0 && flags[2]!=1){
		flags[2]=1;
		qsort(cFinal,(temp+num_of_cs-1),sizeof(struct dimens),cmpfuncy);
		struct dimens *cToComm = (struct dimens *)malloc(sizeof(struct dimens));
		i=0;
		l=0;
		while (cFinal[i+1].y_dim<=gridBoxesPerProcess[0].y_dim+y_dist){
			i++;
			cToComm=(struct dimens *)realloc(cToComm, i*sizeof(struct dimens));
			cToComm[l]=cFinal[i+1];
			l++;
		}
		nums[2]=i-3;
		MPI_Isend(&nums[2],1,MPI_INT,rank-na,3+(10*rank),MPI_COMM_WORLD,&reqs[8]);
		MPI_Isend(&cToComm[0],i-3,structtype,rank-na,3+(10*rank),MPI_COMM_WORLD,&reqs[2]);
		printf("Rank=%d. Send to rank=%d\n",rank, rank-na);
	}
	if (gridBoxesPerProcess[0].y_dim+y_dist+max_dist_y<=1 && flags[3]!=1){
		flags[3]=1;
		qsort(cFinal,(temp+num_of_cs-1),sizeof(struct dimens),cmpfuncy);
		struct dimens *cToComm = (struct dimens *)malloc(sizeof(struct dimens));
		i=0;
		int point=-1;
		for(;;){
			if (cFinal[i].y_dim>=gridBoxesPerProcess[(n*m)-1].y_dim){
				point=i;
				break;
			}
			i++;
		}
		i=1;
		while (cFinal[point+i-1].y_dim<=gridBoxesPerProcess[(n*m)-1].y_dim+y_dist && (point+i-1)<=temp+num_of_cs-2){
			cToComm[i-1]=cFinal[point+i-1];
			i++;
			cToComm=(struct dimens *)realloc(cToComm, i*sizeof(struct dimens));
		}
		nums[3]=i-1;
		MPI_Isend(&nums[3],1,MPI_INT,rank+na,4+(10*rank),MPI_COMM_WORLD,&reqs[9]);
		MPI_Isend(&cToComm[0],i-1,structtype,rank+na,4+(10*rank),MPI_COMM_WORLD,&reqs[3]);
		printf("Rank=%d. Send to rank=%d\n",rank, rank+na);
	}
	if (gridBoxesPerProcess[0].z_dim-max_dist_z>=0 && flags[4]!=1){
		flags[4]=1;
		qsort(cFinal,(temp+num_of_cs-1),sizeof(struct dimens),cmpfuncz);
		struct dimens *cToComm = (struct dimens *)malloc(sizeof(struct dimens));
		i=1;
		while (cFinal[i-1].z_dim<=gridBoxesPerProcess[0].z_dim+z_dist){
			cToComm[i-1]=cFinal[i];
			i++;
			cToComm=(struct dimens *)realloc(cToComm, i*sizeof(struct dimens));
		}
		nums[4]=i-1;
		MPI_Isend(&nums[4],1,MPI_INT,rank-(na*ma),5+(10*rank),MPI_COMM_WORLD,&reqs[10]);
		MPI_Isend(&cToComm[0],i-1,structtype,rank-(na*ma),5+(10*rank),MPI_COMM_WORLD,&reqs[4]);
		printf("Rank=%d. Send to rank=%d\n",rank, rank-(na*ma));
	}
	if (gridBoxesPerProcess[0].z_dim+z_dist+max_dist_z<=1 && flags[5]!=1){
		flags[5]=1;
		qsort(cFinal,(temp+num_of_cs-1),sizeof(struct dimens),cmpfuncz);
		struct dimens *cToComm = (struct dimens *)malloc(sizeof(struct dimens));
		i=0;
		int point=-1;
		for(;;){
			if (cFinal[i].z_dim>=gridBoxesPerProcess[counter-1].z_dim){
				point=i;
				break;
			}
			i++;
		}

		i=1;
		while (cFinal[point+i-1].z_dim<=gridBoxesPerProcess[counter-1].z_dim+z_dist && (point+i-1)<=temp+num_of_cs-2){
			if (cFinal[point+i-1].z_dim<=gridBoxesPerProcess[counter-1].z_dim){
				break;
			}
			cToComm[i-1]=cFinal[point+i-1];
			i++;
			cToComm=(struct dimens *)realloc(cToComm, i*sizeof(struct dimens));
		}
		nums[5]=i-1;
		MPI_Isend(&nums[5],1,MPI_INT,rank+(na*ma),6+(10*rank),MPI_COMM_WORLD,&reqs[11]);
		MPI_Isend(&cToComm[0],i-1,structtype,rank+(na*ma),6+(10*rank),MPI_COMM_WORLD,&reqs[5]);
		printf("Rank=%d. Send to rank=%d\n",rank, rank+(na*ma));
	}
	int re[6];
	double min_c[3];
	double temp_min=-1;
	double min=1000000;
	struct dimens **newArrays;
	newArrays=malloc(6*sizeof(struct dimens*));
	struct dimens *cToRecv;
	qsort(cFinal,(temp+num_of_cs-2),sizeof(struct dimens),cmpfuncx);
	qsort(qFinal,(temp1+num_of_qs-2),sizeof(struct dimens),cmpfuncx);
	for (i=0; i<temp1+num_of_qs-2; i++){
		id=-1;
		for(j=0; j<counter; j++){
			if((gridBoxesPerProcess[j].x_dim<=qFinal[i].x_dim && gridBoxesPerProcess[j].x_dim+x_dist>=qFinal[i].x_dim) 
				&& (gridBoxesPerProcess[j].y_dim<=qFinal[i].y_dim && gridBoxesPerProcess[j].y_dim+y_dist>=qFinal[i].y_dim)
				&& (gridBoxesPerProcess[j].z_dim<=qFinal[i].z_dim && gridBoxesPerProcess[j].z_dim+z_dist>=qFinal[i].z_dim)){
				id=j;
			}
			if (id!=-1){
				break;
			}
		}

		//We first check in Q's box and the nearest ones to avoid more searches
		if ((gridBoxesPerProcess[id].x_dim-x_dist>=min_dist_x && gridBoxesPerProcess[id].y_dim-y_dist>=min_dist_y &&
				gridBoxesPerProcess[id].z_dim-z_dist>=min_dist_z) && (gridBoxesPerProcess[id].x_dim+2*x_dist<=max_dist_x && 
				gridBoxesPerProcess[id].y_dim+2*y_dist<=max_dist_y && gridBoxesPerProcess[id].z_dim+2*z_dist<=max_dist_z)){
		min=1000000;
		temp_min=-1;
		count=0;
		flag=0;
		for(;;){
			//Test if it is in Q's box
			if((gridBoxesPerProcess[id].x_dim<=cFinal[count].x_dim && gridBoxesPerProcess[id].x_dim+x_dist>=cFinal[count].x_dim) 
				&& (gridBoxesPerProcess[id].y_dim<=cFinal[count].y_dim && gridBoxesPerProcess[id].y_dim+y_dist>=cFinal[count].y_dim)
				&& (gridBoxesPerProcess[id].z_dim<=cFinal[count].z_dim && gridBoxesPerProcess[id].z_dim+z_dist>=cFinal[count].z_dim)){
				temp_min=sqrt(pow((qFinal[i].x_dim - cFinal[count].x_dim),2)+pow((qFinal[i].y_dim - cFinal[count].y_dim),2)+
				pow((qFinal[i].z_dim - cFinal[count].z_dim),2));
				if (temp_min<min){
					min=temp_min;
					min_c[0]=cFinal[count].x_dim;
					min_c[1]=cFinal[count].y_dim;
					min_c[2]=cFinal[count].z_dim;
				}
			}
			if (cFinal[count].x_dim>gridBoxesPerProcess[id].x_dim+x_dist){
				if ((sqrt(pow((qFinal[i].x_dim-min_c[0]),2))+pow((qFinal[i].y_dim-min_c[1]),2)+pow((qFinal[i].z_dim-min_c[2]),2))<=
					(sqrt(pow((qFinal[i].x_dim-gridBoxesPerProcess[id].x_dim),2)+pow((qFinal[i].y_dim-gridBoxesPerProcess[id].y_dim),2)
					+pow((qFinal[i].z_dim-gridBoxesPerProcess[id].z_dim),2)))){
						flag=1;
					break;
				}
				break;
			}
			count++;
		}
		//Check to the 26 nearest boxes
		if(flag!=1){
			temp_min=-1;
			min=1000000;
			count=0;
			while (count<temp+num_of_cs-2){

				if((cFinal[count].x_dim>=gridBoxesPerProcess[id].x_dim-x_dist && cFinal[count].y_dim
					>=gridBoxesPerProcess[id].y_dim-y_dist && cFinal[count].z_dim>=gridBoxesPerProcess[id].z_dim-z_dist) &&
					(cFinal[count].x_dim<=gridBoxesPerProcess[id].x_dim+2*x_dist && cFinal[count].y_dim
					<=gridBoxesPerProcess[id].y_dim+2*y_dist && cFinal[count].z_dim<=gridBoxesPerProcess[id].z_dim+2*z_dist))
					{
						temp_min=sqrt(pow((qFinal[i].x_dim - cFinal[count].x_dim),2)+pow((qFinal[i].y_dim - cFinal[count].y_dim),2)+
						pow((qFinal[i].z_dim - cFinal[count].z_dim),2));
						if (temp_min<min){
							min=temp_min;
							min_c[0]=cFinal[count].x_dim;
							min_c[1]=cFinal[count].y_dim;
							min_c[2]=cFinal[count].z_dim;
						}
					}
					if (cFinal[count].x_dim>gridBoxesPerProcess[id].x_dim+x_dist){
						break;
					}
					count++;
				}
			}
		}
		else{
			//Begin to receive if it is of need. The printfs can guarantee that each process' elements 
			//sent, are also received.
			MPI_Barrier(MPI_COMM_WORLD);
			if (gridBoxesPerProcess[0].x_dim-max_dist_x>=0){
				if (flags[6]!=1){
					flags[6]=1;
					MPI_Irecv(&re[0],1,MPI_INT,rank-1,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[12]);
					MPI_Wait(&reqs[12],&stats[0]);
					cToRecv = (struct dimens *)malloc(re[0]*sizeof(struct dimens));
					MPI_Irecv(&cToRecv[0],re[0],structtype,rank-1,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[13]);
				}
				if (flags[6]==1){
					if (waitFlags[0]!=1){
						MPI_Wait(&reqs[13],&stats[6]);
						printf("Rank=%d. Recieved from rank=%d\n",rank, rank-1);
						newArrays[0]=cToRecv;
						waitFlags[0]=1;
					}
					count=0;
					double temp_min=-1;
					double min=1000000;
					for(;;){
						if (cFinal[count].x_dim<=gridBoxesPerProcess[0].x_dim+2*x_dist && cFinal[count].z_dim<=
						gridBoxesPerProcess[0].z_dim+2*z_dist && gridBoxesPerProcess[0].z_dim+2*z_dist<=max_dist_z &&
						cFinal[count].x_dim!=0){

							temp_min=sqrt(pow((qFinal[i].x_dim - cFinal[count].x_dim),2)+pow((qFinal[i].y_dim - cFinal[count].y_dim),2)+
							pow((qFinal[i].z_dim - cFinal[count].z_dim),2));
							if (temp_min<min){
								min=temp_min;
								min_c[0]=cFinal[count].x_dim;
								min_c[1]=cFinal[count].y_dim;
								min_c[2]=cFinal[count].z_dim;
							}
							count++;
						}
						if (cFinal[count].x_dim>gridBoxesPerProcess[0].x_dim+2*x_dist){
							break;
						}
					}
					count=0;
					for(;;){
						if (newArrays[0][count].z_dim<=gridBoxesPerProcess[id].z_dim+2*z_dist){
							temp_min=sqrt(pow((qFinal[i].x_dim - newArrays[0][count].x_dim),2)+pow((qFinal[i].y_dim - newArrays[0][count].y_dim),2)+
							pow((qFinal[i].z_dim - newArrays[0][count].z_dim),2));
							if (temp_min<min){
								min=temp_min;
								min_c[0]=newArrays[0][count].x_dim;
								min_c[1]=newArrays[0][count].y_dim;
								min_c[2]=newArrays[0][count].z_dim;
							}
							count++;
						}
					}
				}
				if (times==0) printf("Rank %d, terminated the 0th work that received from rank %d\n", rank,rank-1);
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (gridBoxesPerProcess[0].x_dim+x_dist+max_dist_x<=1){
				if (flags[7]!=1){
					flags[7]=1;
					MPI_Irecv(&re[1],1,MPI_INT,rank+1,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[14]);
					MPI_Wait(&reqs[14],&stats[1]);
					cToRecv = (struct dimens *)malloc(re[1]*sizeof(struct dimens));
					MPI_Irecv(&cToRecv[0],re[1],structtype,rank+1,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[15]);
				}
				if (flags[7]==1){
					if (waitFlags[1]!=1){
						MPI_Wait(&reqs[15],&stats[7]);
						printf("Rank=%d. Recieved from rank=%d\n",rank, rank+1);
						newArrays[1]=cToRecv;
						waitFlags[1]=1;
					}
					count=0;
					temp_min=-1;
					min=1000000;
					for(;;){
						if (cFinal[count].x_dim<=max_dist_x-2*x_dist && cFinal[count].x_dim!=0){

							temp_min=sqrt(pow((qFinal[i].x_dim - cFinal[count].x_dim),2)+pow((qFinal[i].y_dim - cFinal[count].y_dim),2)+
							pow((qFinal[i].z_dim - cFinal[count].z_dim),2));
							if (temp_min<min){
								min=temp_min;
								min_c[0]=cFinal[count].x_dim;
								min_c[1]=cFinal[count].y_dim;
								min_c[2]=cFinal[count].z_dim;
							}
							count++;
						}
						if (count>=temp+num_of_cs-1){
							break;
						}
					}
					count=0;
					for(;;){
						if (newArrays[1][count].z_dim<=gridBoxesPerProcess[id].z_dim+2*z_dist){
							temp_min=sqrt(pow((qFinal[i].x_dim - newArrays[1][count].x_dim),2)+pow((qFinal[i].y_dim - newArrays[1][count].y_dim),2)+
							pow((qFinal[i].z_dim - newArrays[1][count].z_dim),2));
							if (temp_min<min){
								min=temp_min;
								min_c[0]=newArrays[1][count].x_dim;
								min_c[1]=newArrays[1][count].y_dim;
								min_c[2]=newArrays[1][count].z_dim;
							}
						}
						count++;
						if (count>=re[1]) break;
					}
				}
				if (times==0) printf("Rank %d, terminated the 1st work that received from rank %d\n", rank,rank+1);
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (gridBoxesPerProcess[0].y_dim-max_dist_y>=0){
				if (flags[8]!=1){
					flags[8]=1;
					MPI_Irecv(&re[2],1,MPI_INT,rank-na,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[16]);
					MPI_Wait(&reqs[16],&stats[2]);
					cToRecv = (struct dimens *)malloc(re[2]*sizeof(struct dimens));
					MPI_Irecv(&cToRecv[0],re[2],structtype,rank-na,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[17]);
					qsort(cFinal,(temp+num_of_cs-1),sizeof(struct dimens),cmpfuncy);
				}
				if (flags[8]==1){
					if (waitFlags[2]!=1){
						MPI_Wait(&reqs[17],&stats[8]);
						printf("Rank=%d. Recieved from rank=%d\n",rank, rank-na);
						newArrays[2]=cToRecv;
						waitFlags[2]=1;
					}

					count=0;
					temp_min=-1;
					min=1000000;
					for(;;){

						if (cFinal[count].y_dim<=gridBoxesPerProcess[0].y_dim+2*y_dist && cFinal[count].y_dim!=0){
							temp_min=sqrt(pow((qFinal[i].x_dim - cFinal[count].x_dim),2)+pow((qFinal[i].y_dim - cFinal[count].y_dim),2)+
							pow((qFinal[i].z_dim - cFinal[count].z_dim),2));
							if (temp_min<min ){
								min=temp_min;
								min_c[0]=cFinal[count].x_dim;
								min_c[1]=cFinal[count].y_dim;
								min_c[2]=cFinal[count].z_dim;
							}
						}
						count++;
						if (cFinal[count].y_dim>gridBoxesPerProcess[0].y_dim+2*y_dist){
							break;
						}
					}
					count=0;
					while(count<re[2]){
							temp_min=sqrt(pow((qFinal[i].x_dim - newArrays[2][count].x_dim),2)+pow((qFinal[i].y_dim - newArrays[2][count].y_dim),2)+
							pow((qFinal[i].z_dim - newArrays[2][count].z_dim),2));
							if (temp_min<min){
								min=temp_min;
								min_c[0]=newArrays[2][count].x_dim;
								min_c[1]=newArrays[2][count].y_dim;
								min_c[2]=newArrays[2][count].z_dim;
							}
							count++;
					}
				}
				if (times==0) printf("Rank %d, terminated the 2nd work that received from rank %d\n", rank,rank-na);

			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (gridBoxesPerProcess[0].y_dim+y_dist+max_dist_y<=1){
				if (flags[9]!=1){
					flags[9]=1;
					MPI_Irecv(&re[3],1,MPI_INT,rank+na,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[18]);
					MPI_Wait(&reqs[18],&stats[3]);
					cToRecv = (struct dimens *)malloc(re[3]*sizeof(struct dimens));
					MPI_Irecv(&cToRecv[0],re[3],structtype,rank+na,MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[19]);
					qsort(cFinal,(temp+num_of_cs-2),sizeof(struct dimens),cmpfuncy);
					for (l=0; l<re[3]; l++){
						if (cToRecv[l].y_dim==0 && cToRecv[l].x_dim==0){
						}
					}
				}
				if (flags[9]==1){
					if (waitFlags[3]!=1){
						MPI_Wait(&reqs[19],&stats[9]);
						printf("Rank=%d. Recieved from rank=%d\n",rank, rank+na);
						newArrays[3]=cToRecv;
						waitFlags[3]=1;
					}
					count=0;
					temp_min=-1;
					min=1000000;
					for(;;){
						if (cFinal[count].y_dim<=max_dist_y-2*y_dist && cFinal[count].y_dim!=0){
							temp_min=sqrt(pow((qFinal[i].x_dim - cFinal[count].x_dim),2)+pow((qFinal[i].y_dim - cFinal[count].y_dim),2)+
							pow((qFinal[i].z_dim - cFinal[count].z_dim),2));
							if (temp_min<min){
								min=temp_min;
								min_c[0]=cFinal[count].x_dim;
								min_c[1]=cFinal[count].y_dim;
								min_c[2]=cFinal[count].z_dim;
							}
						}
						count++;
						if (count>=temp+num_of_cs-1){
							break;
						}
					}
					count=0;
					for(;;){
						if (newArrays[3][count].x_dim<=gridBoxesPerProcess[id].x_dim+2*x_dist){
							temp_min=sqrt(pow((qFinal[i].x_dim - newArrays[3][count].x_dim),2)+pow((qFinal[i].y_dim - newArrays[3][count].y_dim),2)+
							pow((qFinal[i].z_dim - newArrays[3][count].z_dim),2));
							if (temp_min<min){
								min=temp_min;
								min_c[0]=newArrays[3][count].x_dim;
								min_c[1]=newArrays[3][count].y_dim;
								min_c[2]=newArrays[3][count].z_dim;
							}
						}
						count++;
						if (count>=re[3]) break;
					}
				}
				if (times==0) printf("Rank %d, terminated the 3rd work that received from rank %d\n", rank,rank+na);
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (gridBoxesPerProcess[0].z_dim-max_dist_z>=0){
				if (flags[10]!=1){
					flags[10]=1;
					MPI_Irecv(&re[4],1,MPI_INT,rank-(na*ma),MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[20]);
					MPI_Wait(&reqs[20],&stats[4]);
					cToRecv = (struct dimens *)malloc(re[4]*sizeof(struct dimens));
					MPI_Irecv(&cToRecv[0],re[4],structtype,rank-(na*ma),MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[21]);
					qsort(cFinal,(temp+num_of_cs-2),sizeof(struct dimens),cmpfuncz);
				}
				if (flags[10]==1){
					if (waitFlags[4]!=1){
						printf("Rank%d. Waiting to receive from rank=%d\n",rank, rank-(na*ma));
						MPI_Wait(&reqs[21],&stats[10]);
						printf("Rank=%d. Recieved from rank=%d\n",rank, rank-(na*ma));
						newArrays[4]=cToRecv;
						waitFlags[4]=1;
					}
					count=0;
					temp_min=-1;
					min=1000000;
					for(;;){
						if (cFinal[count].z_dim<=gridBoxesPerProcess[0].z_dim+2*z_dist){

							temp_min=sqrt(pow((qFinal[i].x_dim - cFinal[count].x_dim),2)+pow((qFinal[i].y_dim - cFinal[count].y_dim),2)+
							pow((qFinal[i].z_dim - cFinal[count].z_dim),2));
							if (temp_min<min){
								min=temp_min;
								min_c[0]=cFinal[count].x_dim;
								min_c[1]=cFinal[count].y_dim;
								min_c[2]=cFinal[count].z_dim;
							}
						}
						count++;
						if (cFinal[count].z_dim>gridBoxesPerProcess[0].z_dim+2*z_dist){
							break;
						}
					}
					count=0;
					while(count<re[4]){
							temp_min=sqrt(pow((qFinal[i].x_dim - newArrays[4][count].x_dim),2)+pow((qFinal[i].y_dim - newArrays[4][count].y_dim),2)+
							pow((qFinal[i].z_dim - newArrays[4][count].z_dim),2));
							if (temp_min<min){
								min=temp_min;
								min_c[0]=newArrays[4][count].x_dim;
								min_c[1]=newArrays[4][count].y_dim;
								min_c[2]=newArrays[4][count].z_dim;
							}
							count++;
					}
				}
				if (times==0) printf("Rank %d, terminated the 4th work that received from rank %d\n", rank,rank-(ma*na));
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (gridBoxesPerProcess[0].z_dim+z_dist+max_dist_z<=1){
				if (flags[11]!=1){
					flags[11]=1;
					MPI_Irecv(&re[5],1,MPI_INT,rank+(na*ma),MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[22]);
					MPI_Wait(&reqs[22],&stats[5]);
					cToRecv = (struct dimens *)malloc(re[5]*sizeof(struct dimens));
					MPI_Irecv(&cToRecv[0],re[5],structtype,rank+(na*ma),MPI_ANY_TAG,MPI_COMM_WORLD,&reqs[23]);
					qsort(cFinal,(temp+num_of_cs-2),sizeof(struct dimens),cmpfuncz);
				}
				if (flags[11]==1){
					if (waitFlags[5]!=1){
						printf("Rank%d. Waiting to receive from rank=%d\n",rank, rank+(na*ma));
						MPI_Wait(&reqs[23],&stats[11]);
						printf("Rank%d. Recieved from rank=%d\n",rank, rank+(na*ma));
						newArrays[5]=cToRecv;
						waitFlags[5]=1;
					}
					count=0;
					temp_min=-1;
					min=1000000;
					for(;;){
						if (cFinal[count].z_dim<=max_dist_z-2*z_dist &&
						 cFinal[count].z_dim!=0){

							temp_min=sqrt(pow((qFinal[i].x_dim - cFinal[count].x_dim),2)+pow((qFinal[i].y_dim - cFinal[count].y_dim),2)+
							pow((qFinal[i].z_dim - cFinal[count].z_dim),2));
							if (temp_min<min){
								min=temp_min;
								min_c[0]=cFinal[count].x_dim;
								min_c[1]=cFinal[count].y_dim;
								min_c[2]=cFinal[count].z_dim;
							}
						}
						count++;
						if (count>=temp+num_of_cs-1){
							break;
						}
					}
					count=0;
					for(;;){
							temp_min=sqrt(pow((qFinal[i].x_dim - newArrays[5][count].x_dim),2)+pow((qFinal[i].y_dim - newArrays[5][count].y_dim),2)+
							pow((qFinal[i].z_dim - newArrays[5][count].z_dim),2));
							if (temp_min<min){
								min=temp_min;
								min_c[0]=newArrays[5][count].x_dim;
								min_c[1]=newArrays[5][count].y_dim;
								min_c[2]=newArrays[5][count].z_dim;
							}
						count++;
						if (count>=re[5]) {
							break;
						}
				}
			}
			if (times==0) printf("Rank %d, terminated the 5th work that received from rank %d\n", rank,rank+(ma*na));
		}
		}
		times=1;
	}
		//End of search

	//Give as output the times.
	
 	gettimeofday (&endwtime, NULL);

 	double time2[size];
 	double seq_time2[size];
 	 seq_time2[rank] = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		     + endwtime.tv_sec - startwtime.tv_sec);
 	MPI_Allgather(&seq_time2[rank],1,MPI_DOUBLE,&time2[0],1,MPI_DOUBLE,MPI_COMM_WORLD);

 	double max_time1=-1,min_time1=1000000;
 	int max_point1,min_point1;
 	for (i=0; i<size; i++){
 		if (time2[i]<=min_time1){
 			min_time1=time2[i];
 			min_point1=i;
 		}
 		if (time2[i]>=max_time1){
 			max_time1=time2[i];
 			max_point1=i;
 		}
 	}


 	if (rank==0){
 		printf("Number of Cs and Qs=%d", NQ);
 		printf("Number of process P=%d\n", P);
 		printf("Number of dimensions nxmxk=\n", atoi(argv[3]));
 		printf("/--------------------------------------------------/\n");
 		printf("/--------------------------------------------------/\n");
 		printf("Generating Numbers and QuickShorting (QUICKEST PROCESS)= %f\n",time1[min_point]);
	  	printf("Generating Numbers and QuickShorting (SLOWEST PROCESS)= %f\n\n",time1[max_point]);
 		printf("Complete searching (QUICKEST PROCESS)= %f\n", time2[min_point1]);
 		printf("Complete searching (SLOWEST PROCESS)= %f\n", time2[max_point1]);
 	}


	MPI_Finalize();

}

