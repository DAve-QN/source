#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mkl.h>
#include <sys/time.h>

int cols;
double val, x;
char* line;
int max_line_len = 1024*10;

// helper function
void printArray(double *arr, int n){
    int i;
    for (i = 0; i < n; ++i)
    {
        printf("%.10f\n", arr[i] );
    }

}

void getMaxMin(int *colidx, double *vals, int d, int nnz, int m, double *max, double *min, int *nnzArray){
    int i;
    for (i = 0; i < d; ++i)
    {
        max[i]=-1e9;
        min[i]=1e9;
        // what if there are 0's in the column but we don't
        // see it since it is sparse?!
        // we have to consider 0's too
        if(nnzArray[i]<m){
            max[i]=0;
            min[i]=0;
        }
    }

    for (i = 0; i < nnz; ++i)
    {
        int index = colidx[i]-1;
        double val = vals[i];
        if (max[index]<val)
        {
            max[index]= val;
        }
        if (min[index]>val)
        {
            min[index]=val;
        }

    }
    for ( i = 0; i < d; ++i)
    {
        if(max[i]< -1e8)
            max[i]=0.0;
        if(min[i]>1e8)
            min[i]=0.0;
    }
}

// normalize the dataset between 0 and 1
void normalize(int *colidx, double *vals, int d, int nnz, double* max, double *min){
    int i;

    for (i = 0; i < nnz; ++i)
    {
        int index = colidx[i]-1;
        
        if (max[index]-min[index]!=0)
        {
            vals[i] = (vals[i]-min[index])/(max[index]-min[index]);
        }else{
           vals[i] = (vals[i]-min[index]); 
        }
       
    }

}



//  s=x-y
void xpy(double *x, double *y, double *s, int d){
            memset(s,0,sizeof(double)*d);
            cblas_daxpy (d, -1, y, 1, s, 1);
            cblas_daxpy (d, 1, x, 1, s, 1);
}

// the gradient function for logistic regression
void gradfun(double *x,
    double *vals,
    int *colidx,
    int *rowidx,
    char *descr ,
    double *y ,
     int d,
     int m,
     int m_local,
     double lambda,
     double *g){

    double alpha = 1.0;
    double nalpha = -1.0;
    double zero =0.0;
    int inc = 1;
    int i;

    double *Ax = (double*) malloc (m_local*sizeof(double));
    double *v = (double*) malloc (m_local*sizeof(double));
    double *e = (double*) malloc (m_local*sizeof(double));
    double *ep1 = (double*) malloc (m_local*sizeof(double));
    
    // op: Ax = A*x
    char trans ='N';
    mkl_dcsrmv (&trans
        , &m_local
         , &d
          , &alpha
           , descr
            , vals
             , colidx
              , rowidx
               , rowidx+1
                , x
                 , &zero
                  , Ax );

    // op: v = Ax.*y
    vdMul( m_local, Ax, y, v );

    // op: e = exp(v)
    vdExp( m_local, v, e );

    // op: ep1 = 1/(1+e)
    for (i = 0; i < m_local; ++i)
    {
        ep1[i] = 1.0/(1 + e[i]);
    }

    //op: v = y.*ep1
    vdMul( m_local, y, ep1, v );
    
    
    // op: g = -A*vs
    char transa ='T';
    mkl_dcsrmv (&transa 
        , &m_local
         , &d
          , &nalpha
           , descr
            , vals
             , colidx
              , rowidx
               , rowidx+1
                , v
                 , &zero
                  , g );



    // double a = (1.0*m_local)/m*lambda;
    // double b = 1.0/m;
    double a = lambda;
    double b = 1.0/m_local;

    cblas_daxpby (d, a, x, inc, b, g, inc);
    //printArray(g,d);

    free(v);
    free(e);
    free(ep1);
    free(Ax);

}

// objective function for logistic regression
double objective_fun(double *x,
    double *vals,
    int *colidx,
    int *rowidx,
    char *descr, double *y , int d, int m, double lambda){

    double alpha = 1.0;
    double nalpha = -1.0;
    double zero =0.0;
    int inc = 1;
    int i;

    
    double *nAx = (double*) malloc (m*sizeof(double));
    double *v = (double*) malloc (m*sizeof(double));
    double *e = (double*) malloc (m*sizeof(double));

    // op: nAx = -A*x
    char transa ='N';
    mkl_dcsrmv (&transa 
        , &m
         , &d
          , &nalpha
           , descr
            , vals
             , colidx
              , rowidx
               , rowidx+1
                , x
                 , &zero
                  , nAx );

    // op: v = nAx.*y
    vdMul( m, nAx, y, v );

    // op: e = exp(v)
    vdExp( m, v, e );

    // op: v = ln(e+1)
    vdLog1p( m, e, v );

    double sum = 0;
    for (i = 0; i < m; ++i)
    {
        sum+= v[i];
    }
    sum = sum/m;
    sum += lambda/2*ddot(&d, x, &inc, x, &inc);


    free(v);
    free(e);
    free(nAx);

    return sum;
}


// helper function to read dataset
static char* readline(FILE *input)
{
  int len;
  if(fgets(line,max_line_len,input) == NULL)
    return NULL;

  while(strrchr(line,'\n') == NULL)
    {
      max_line_len *= 2;
      line = (char *) realloc(line,max_line_len);
      len = (int) strlen(line);
      if(fgets(line+len,max_line_len-len,input) == NULL)
        break;
    }
  return line;
}

void readgg(char* fname, int *rowidx,int *colidx,
    double *vals,double *y , int *nnz_local, int *m_local, int* nnzArray)
{
    
    int i;
    FILE * file;
    file = fopen(fname, "r");
    printf("%s\n",fname );


    if(file == NULL){
        printf("File not found!\n");
        return;
    }

    line= (char*) malloc(max_line_len*sizeof(char));

    int count_rowidx=0,count_colidx=0;;
    rowidx[count_rowidx]=1;
    i = 0;
    
    
      while (1)
    {
      if(readline(file)==NULL){
        //printf("NULL pointer\n");
        break;
      }

      char *label, *value, *index, *endptr;
      label = strtok(line," \t\n");
      x = strtod(label, &endptr);
      while(1)
        {
          index = strtok(NULL,":");
          
          value = strtok(NULL," \t");

          if(value == NULL){
            break;
          }

          cols = (int) strtol(index,&endptr,10);
          colidx[count_colidx]=cols;
          nnzArray[cols-1]++;

          val =  strtod(value, &endptr);
          vals[count_colidx]=val;
        count_colidx++;
          i++;
        }
      count_rowidx++;
      rowidx[count_rowidx]=i+1;
      y[count_rowidx-1]=x;

    }
    *nnz_local = count_colidx;
    *m_local = count_rowidx;
  fclose(file);
}


int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);
    // Initialize the MPI environment
    
    int world_size;
    int rank;

    struct timeval start,end;

    // Get the number of processes and rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // Print off a hello world message
    printf("rank %d: started\n", rank);
    
    if(world_size<2){
        printf("%s\n", "There should be at least two processors!");
        MPI_Finalize();
        return 0;
    }


// INPUT
    if(argc<9){
        printf("Input Format: pathname #rows #nnzs #cols #iterations lambda gamma freq\n");
        MPI_Finalize();
        return 0;
    }
    char* pathName = argv[1];
    int m = atoi(argv[2]);
    int nnz = atoi(argv[3]);
    long long d = atoll(argv[4]);
    int Iter = atoi(argv[5]);
    double lambda = atof(argv[6]);
    double eta = atof(argv[7]);
    int freq = atoi(argv[8]);

   // MPI_Barrier(MPI_COMM_WORLD);

    if(rank==0){
        printf("lambda is %f\n",lambda );
        printf("Iter is %d\n",Iter );
        printf("eta is %f\n",eta );
    }
    
    //char* fileName = argv[2];
    //int partition = atoi(argv[3]);

        double one = 1.0;
        MKL_INT inc = 1;
        double zero = 0.0;
        char trans = 'N';
        int cumX_size = (Iter)/freq +20;

    double *cumX = (double *) malloc((cumX_size*d)*sizeof(double));
    long *times = (long*) malloc((cumX_size)*sizeof(long));

    int *rowidx, *colidx, *nnzArray, nnz_local, m_local; // m_local: number of samples for each processor
    double *vals,*y;
    char *descrA;


    rowidx=(int *)malloc((m+1)*sizeof(int));
    colidx=(int *)malloc(nnz*sizeof(int));
    vals=(double *)malloc(nnz*sizeof(double));
    y=(double *)malloc(m*sizeof(double));
    
    nnzArray=(int *)malloc(d*sizeof(int));
    descrA=(char *)malloc(6*sizeof(char));
    memset(nnzArray,0,d*sizeof(int));

    // Descriptor of main sparse matrix properties
    descrA[0]='G';
    descrA[3]='F';
    

    // local gradients : g
    // local gradient at xold: gold
    // reduced gradient on root: gsum
    double *g = (double*) malloc (d*sizeof(double));
    double *gold = (double*) malloc (d*sizeof(double));
    double *gsum = (double*) malloc (d*sizeof(double));
    memset(g, 0, sizeof(double)*d);
    memset(gold, 0, sizeof(double)*d);
    memset(gsum, 0, sizeof(double)*d);


    //Initialization
    double *x = (double*) malloc (d*sizeof(double));
    double *u = (double*) malloc (d*sizeof(double));
    double *B =(double*) malloc (d*d*sizeof(double));
    double *xold = (double*) malloc (d*sizeof(double));

    memset(u, 0, sizeof(double)*d);
    memset(x, 0, sizeof(double)*d);
    memset(xold, 0, sizeof(double)*d);
    
    
    // max and min of the dataset for normalization    
    double *max = (double *) malloc(d*sizeof(double));
    double *min = (double *) malloc(d*sizeof(double));
    int i;
    for ( i = 0; i < d; ++i)
    {
        max[i]=-1e9;
        min[i]=1e9;
    }
    double *maxAll = (double *) malloc(d*sizeof(double));
    double *minAll = (double *) malloc(d*sizeof(double));
    if(rank>0){

        long long j = 0;
        for ( j = 0; j < d*d; ++j)
        {
            B[j]=0;
        }

        for ( j = 0; j < d; ++j)
        {
            B[j*(d+1)] = 1.0;
        }


        char buf[100];
        strcat(pathName, "-");
        sprintf(buf, "%d", rank-1);
        strcat(pathName,buf);

        readgg(pathName, rowidx, colidx, vals, y, &nnz_local, &m_local, nnzArray);
        // normalizing the matrix between 0 and 1
        // first we get the max and min of local dataset
        // then we reduce it to get the overall max and min
        
        getMaxMin(colidx, vals, d, nnz_local, m_local, max, min, nnzArray);

    }
    
    // get the maximum and minimum over all processors
    MPI_Allreduce(min, minAll, d, MPI_DOUBLE, MPI_MIN,
           MPI_COMM_WORLD);
    MPI_Allreduce(max, maxAll, d, MPI_DOUBLE, MPI_MAX,
           MPI_COMM_WORLD);

    if (rank>0)
    {
        //normalize the dataset using maxall and minall
        // which contain the max and min for each column

        normalize(colidx, vals, d, nnz_local, maxAll, minAll);
        
        // initial gradients 
        gradfun(x , vals,colidx,rowidx, descrA, y, d ,m, m_local,lambda, g);

        gradfun(xold , vals,colidx,rowidx, descrA, y, d ,m,m_local,lambda, gold);
    }
    if(rank == 0 ){

        // just for objective calculation
        // we don't need this for performance evaluation:
        readgg(pathName, rowidx, colidx, vals, y, &nnz_local, &m_local,nnzArray);
        normalize(colidx, vals, d, nnz_local, maxAll, minAll);

        long long j = 0;
        for (j = 0; j < d; ++j)
        {
            B[j*(d+1)] = 1.0/(world_size-1);
        }
    }

    MPI_Allreduce(g, gsum, d, MPI_DOUBLE, MPI_SUM,
           MPI_COMM_WORLD);

    // new initial condition: which is one step gradeint descent
    double neta= -eta/(world_size-1);
    cblas_daxpy (d, neta, gsum, 1, x, 1);
  
    //start timing
    if(rank ==0){
        memcpy (cumX, x, d*sizeof(double)); 
        gettimeofday(&start,NULL);
        times[0] = 0;
    }
    
    if(rank>0){
        int dim = (int) d;
        double *s =(double*) malloc (d*sizeof(double));
        double *ys =(double*) malloc (d*sizeof(double));
        double *q =(double*) malloc (d*sizeof(double));
        double *u1 =(double*) malloc (d*sizeof(double));
        double *u2 =(double*) malloc (d*sizeof(double));
        double *update =(double*) malloc ((3*d+2)*sizeof(double));
        double alpha =0, beta =0;

        while(1){

            // s= x-xold
            xpy(x, xold, s, d);
            
            // gradinet
            gradfun(x , vals, colidx, rowidx, descrA, y, d ,m,  m_local,lambda, g);
            
            // ys = g - gold
            xpy(g, gold, ys, d);
            
            // q = B*s
            dgemv(&trans, &dim, &dim, &one, B, &dim, s, &inc, &zero, q, &inc);
            
            alpha = ddot(&dim, ys, &inc, s, &inc);
            beta = ddot(&dim, s, &inc, q, &inc);  
            double alpha_inv = 1.0/alpha;
            double beta_ninv = -1.0/beta;

            // u1 = B*xold
            dgemv(&trans, &dim, &dim, &one, B, &dim, xold, &inc, &zero, u1, &inc);
            //printf("rank %d:  u1 done\n",rank);

            //rank 1 update of B > B = B + y'y/alpha - q'q/beta
            dger(&dim, &dim, &alpha_inv, ys, &inc, ys, &inc, B, &dim);
            dger(&dim, &dim, &beta_ninv, q, &inc, q, &inc, B, &dim);

            // u2 = B*x
            dgemv(&trans, &dim, &dim, &one, B, &dim, x, &inc, &zero, u2, &inc);
            //printf("rank %d:  u2 done\n",rank);

            // u = u2-u1
            xpy(u2, u1, u, d);

            // xold = x
            // gold = g
            memcpy (xold, x, d*sizeof(double)); 
            memcpy (gold, g, d*sizeof(double)); 

            // send results to the master
            memcpy (update, u, d*sizeof(double));
            memcpy (update+d, ys, d*sizeof(double)); 
            memcpy (update+2*d, q, d*sizeof(double)); 
            update[3*d] = alpha;
            update[3*d+1] = beta; 
            MPI_Request request;

            MPI_Isend(
                update,
                3*d+2,
                MPI_DOUBLE,
                0,
                0,
                MPI_COMM_WORLD,
                &request);
            MPI_Request_free(&request);
            
            // recieve x
            MPI_Status status;
            MPI_Recv(
                x,
                d,
                MPI_DOUBLE,
                0,
                MPI_ANY_TAG,
                MPI_COMM_WORLD,
                &status);

            if(status.MPI_TAG == 3){
                printf("rank %d: Recieved signal. Finishing the process \n", rank);
                break;
            }            
            
        }

        free(s);
        free(ys);
        free(q);
        free(u1);
        free(u2);
        free(update);
        
    }

    if (rank ==0)
    {   
        int dim = (int) d;
        double *update =(double*) malloc ((3*d+2)*sizeof(double));
        double *v =(double*) malloc (d*sizeof(double));
        double *w =(double*) malloc (d*sizeof(double));
        double *u_gsum =(double*) malloc (d*sizeof(double));


        
        int it = 0;
        while(it < Iter){
            MPI_Status status;
            MPI_Recv(
            update,
            3*d+2,
            MPI_DOUBLE,
            MPI_ANY_SOURCE,
            0,
            MPI_COMM_WORLD,
            &status);

            // u = u+du (du = update[0::d-1])
            cblas_daxpy (d, 1, update, 1, u, 1);

            // gsum = gsum+ys (ys = update[d::2d-1])
            cblas_daxpy (d, 1, update+d, 1, gsum, 1);

            // v = B*ys
            dgemv(&trans, &dim, &dim, &one, B, &dim, update+d, &inc, &zero, v, &inc);
            
            //rank 1 update of B > B = B-vv'/(alpha+v'y)
            // in algorithm, this update is stored in 'U'
            double alpha_up = -1.0/(update[3*d]+ddot(&dim, update+d, &inc, v, &inc));
            dger(&dim, &dim, &alpha_up, v, &inc, v, &inc, B, &dim);

            // w = B*q (U*q) (q = update[2d::3d-1])
            dgemv(&trans, &dim, &dim, &one, B, &dim, update+2*d, &inc, &zero, w, &inc);
            
            //rank 1 update of B > B = B+w*w'/(beta-q'*w);
            double beta_up = 1.0/(update[3*d+1]-ddot(&dim, update+2*d, &inc, w, &inc));
            dger(&dim, &dim, &beta_up, w, &inc, w, &inc, B, &dim);

            // u_gsum = u - gsum
            // update x = B*(u-gsum)
            xpy(u, gsum, u_gsum, d);
            dgemv(&trans, &dim, &dim, &one, B, &dim, u_gsum, &inc, &zero, x, &inc);

            if(it%freq == 0 ){
                gettimeofday(&end,NULL);
                long seconds = end.tv_sec - start.tv_sec;
                times[it/freq +1] = (seconds*1000)+(end.tv_usec - start.tv_usec)/1000;
                memcpy (cumX+(it/freq +1)*d, x, d*sizeof(double)); 
            }
            

            if(it != Iter-1){
                MPI_Send(x, d, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            }else{
                // sending exit signal to all processors
                // exit signal has tag=3
                for ( i = 1; i < world_size; ++i)
                {
                    printf("rank %d: ROOT: sent finish signal to rank %d\n", rank, i );
                    MPI_Send(x, d, MPI_DOUBLE, i, 3, MPI_COMM_WORLD);
                }
            }
            

        it++;
        }
        free(update);
        free(v);
        free(w);
        free(u_gsum);
    }


    // Finalize the MPI environment.
    MPI_Finalize();


    // the next 2 lines are for computing the objective function
    // to report later
    double *gval =(double*) malloc (d*sizeof(double));

    if(rank == 0){
        int dim = (int) d;
        for (i = 0; i < (Iter-1)/freq; ++i)
    {
        double obj_value = objective_fun(cumX+i*d, vals,colidx,rowidx, descrA , y , d, m, lambda);
        gradfun(cumX+i*d , vals,colidx,rowidx, descrA, y, d ,m,  m_local,lambda, gval);
        double grad_value = ddot(&dim, gval, &inc, gval, &inc);
        printf(" %ld , %.8f,%.8f \n", times[i], obj_value,grad_value);
    }
    }

    free(vals);
    free(rowidx);
    free(colidx);
    free(y);
    free(B);
    
    
    return 0;
}