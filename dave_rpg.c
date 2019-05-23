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

void printArray(double *arr, int n){
    int i;
    for (i = 0; i < n; ++i)
    {
        printf("%.10f\n", arr[i] );
    }

}

void getMaxMin(int *colidx, double *vals, int d, int nnz, int m,  double *max, double *min, int *nnzArray){
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
    for (i = 0; i < d; ++i)
    {
        if(max[i]< -1e8)
            max[i]=0.0;
        if(min[i]>1e8)
            min[i]=0.0;
    }
    //printArray(min,d);
}
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
void xpby(double *x, double *y, double *s, double b, int d){
            memset(s,0,sizeof(double)*d);
            cblas_daxpy (d, -b, y, 1, s, 1);
            cblas_daxpy (d, 1, x, 1, s, 1);
}


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
   
    // mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
    //     alpha, csrA, descr, x, zero, Ax);
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
    // mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE,
    //      nalpha, csrA, descr, v, zero, g);
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
    double a = lambda;
    double b = 1.0/m_local;
    cblas_daxpby (d, a, x, inc, b, g, inc);

    free(v);
    free(e);
    free(ep1);
    free(Ax);

}



void gradfunSample(double *x,
    double *vals,
    int *colidx,
    int *pointerB,
    int *pointerE,
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
   
    // mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
    //     alpha, csrA, descr, x, zero, Ax);
    char trans ='N';
    mkl_dcsrmv (&trans
        , &m_local
         , &d
          , &alpha
           , descr
            , vals
             , colidx
              , pointerB
               , pointerE
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
    // mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE,
    //      nalpha, csrA, descr, v, zero, g);
    char transa ='T';
    mkl_dcsrmv (&transa 
        , &m_local
         , &d
          , &nalpha
           , descr
            , vals
             , colidx
              , pointerB
               , pointerE
                , v
                 , &zero
                  , g );
    double a = lambda;
    double b = 1.0/m_local;
    cblas_daxpby (d, a, x, inc, b, g, inc);

    free(v);
    free(e);
    free(ep1);
    free(Ax);

}

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
    // op: v = nAx.*y
    // op: e = exp(v)
    // mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
    //     nalpha, csrA, descr, x, zero, nAx);
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

    vdMul( m, nAx, y, v );
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

double objective_funSample(double *x,
    double *vals,
    int *colidx,
    int *pointerB,
    int *pointerE,
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
    // op: v = nAx.*y
    // op: e = exp(v)
    // mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
    //     nalpha, csrA, descr, x, zero, nAx);
    char transa ='N';
    mkl_dcsrmv (&transa 
        , &m
         , &d
          , &nalpha
           , descr
            , vals
             , colidx
              , pointerB
               , pointerE
                , x
                 , &zero
                  , nAx );

    vdMul( m, nAx, y, v );
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
          // if(cols==NULL){
          //   printf("WTF3\n");
          // }
          //colidx.push_back(cols-1);
          colidx[count_colidx]=cols;
          nnzArray[cols-1]++;
            //printf("%d\n",cols );

          val =  strtod(value, &endptr);
          // if(val==NULL){
          //   printf("WTF4\n");
          // }
          //vals.push_back(val);
          vals[count_colidx]=val;
          //printf("%d\n",count_colidx );
        count_colidx++;
          i++;
        }
      count_rowidx++;
      rowidx[count_rowidx]=i+1;
    //printf("%d\n",count_rowidx );

      y[count_rowidx-1]=x;
    }
    *nnz_local = count_colidx;
    *m_local = count_rowidx;
  fclose(file);
  //printf("DONE\n");
}


int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc,&argv);

    // Get the number of processes and rank
    int world_size;
    int rank;
    struct timeval start,end;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // Print off a hello world message
    printf("rank %d: started\n",
     rank);
    if(world_size<2){
        printf("%s\n", "There should be at least two processors!");
        MPI_Finalize();
        return 0;
    }



// INPUT
    if(argc<10){
        printf("Input Format: pathname nrows nnz ncols iterations lambda eta p [freq of printing]\n");
        MPI_Finalize();
        return 0;
    }
    char* pathName = argv[1];
    int m = atoi(argv[2]);
    int nnz = atoi(argv[3]);
    int d = atoi(argv[4]);
    int Iter = atoi(argv[5]);
    double lambda = atof(argv[6]);
    double eta = atof(argv[7]);
    int p = atoi(argv[8]);
    int freq = atoi(argv[9]);
    //char* fileName = argv[2];
    //int partition = atoi(argv[3]);
    if(rank==0){
        printf("lambda is %f\n",lambda );
        printf("Iter is %d\n",Iter );
        printf("eta is %f\n",eta );
        printf("p is %d\n",p );
    }
        double one = 1.0;
        int j;
        MKL_INT inc = 1;
        double zero = 0.0;
        char trans = 'N';
        int cumX_size = (Iter)/freq +20;

        double L = 0.5; // initial Lipschitz constant
        int seed = 42; // seed for line search method sampling
        srand(seed);

    double *cumX = (double *) malloc((cumX_size*d)*sizeof(double));
    long *times = (long*) malloc((cumX_size)*sizeof(long));

    int *rowidx,*colidx, *nnzArray, nnz_local, m_local; // m_local: number of samples for each processor
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
    // struct matrix_descr descrA;
    // descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    descrA[0]='G';
    descrA[3]='F';

    // local gradients : g
    // reduced gradient on root: gsum
    double *g = (double*) malloc (d*sizeof(double));
    double *gsum = (double*) malloc (d*sizeof(double));
    memset(g, 0, sizeof(double)*d);
    memset(gsum, 0, sizeof(double)*d);
    // handle to input matrix
    // sparse_matrix_t       csrA;

    //Initialization
    double *x = (double*) malloc (d*sizeof(double));
    memset(x, 0, sizeof(double)*d);

    
    
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

        char buf[100];
        strcat(pathName, "-");
        sprintf(buf, "%d", rank-1);
        strcat(pathName,buf);
        
        readgg(pathName, rowidx, colidx, vals, y, &nnz_local, &m_local, nnzArray);
        // Create handle with matrix stored in CSR format
        
        // normalizing the matrix between 0 and 1
        // first we get the max and min of local dataset
        // then we reduce it to get the overall max and min
        
        getMaxMin(colidx, vals, d, nnz_local, m_local, max, min,nnzArray);

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

        // mkl_sparse_d_create_csr ( &csrA, SPARSE_INDEX_BASE_ONE,
        //                             m_local,  // number of rows
        //                             d,  // number of cols
        //                             rowidx,
        //                             rowidx+1,
        //                             colidx,
        //                             vals );
        // initial gradients 
        gradfun(x , vals,colidx,rowidx, descrA, y, d ,m, m_local,lambda, g);

    }
    if(rank == 0 ){

        // just for objective calculation
        // we don't need this for performance evaluation:
       readgg(pathName, rowidx, colidx, vals, y, &nnz_local, &m_local,nnzArray);
        normalize(colidx, vals, d, nnz_local, maxAll, minAll);
        // Create handle with matrix stored in CSR format
        // mkl_sparse_d_create_csr ( &csrA, SPARSE_INDEX_BASE_ONE,
        //                             m_local,  // number of rows
        //                             d,  // number of cols
        //                             rowidx,
        //                             rowidx+1,
        //                             colidx,
        //                             vals );
    
    }

    MPI_Allreduce(g, gsum, d, MPI_DOUBLE, MPI_SUM,
           MPI_COMM_WORLD);

    // new initial condition: which is one step gradeint descent
    // [could be removed most likely]
    double neta= -eta/(world_size-1);
    cblas_daxpy (d, neta, gsum, 1, x, 1);
    
    //start timing
    if(rank ==0){
        memcpy (cumX, x, d*sizeof(double)); 
        gettimeofday(&start,NULL);
        times[0] = 0;
    }

    if(rank>0){
        double *delta =(double*) malloc (d*sizeof(double));
        double *z =(double*) malloc (d*sizeof(double));
        double *xplus =(double*) malloc (d*sizeof(double));
        double *xplus_x =(double*) malloc (d*sizeof(double));
        double *xbar =(double*) malloc (d*sizeof(double));
            
        // Line search vectors and initilization
        // b_size: the batch size for line search method [default =1 ]
        int b_size = 1;
        double *gLineSearch = (double*) malloc (d*sizeof(double));
        int *pointerE = (int*) malloc (b_size*sizeof(int));
        int *pointerB = (int*) malloc (b_size*sizeof(int));
        double *var1 =(double*) malloc (d*sizeof(double)); //temp variable
        double obj_valueKplus = 1;
        double obj_valueK = 0 ;

        while(1){

            memset(delta, 0, d*sizeof(double));
            memcpy (xbar, x, d*sizeof(double)); 


            // ********** Line search method

            while(obj_valueKplus > obj_valueK){
                // chose a sample
                L=L*2;
                for (j = 0; j < b_size; ++j)
                {
                    int ind = rand()%m_local;
                    pointerB[j] = rowidx[ind];
                    pointerE[j] = rowidx[ind+1];
                }
            
                gradfunSample(xbar , vals, colidx, pointerB,pointerE, descrA, y, d ,m, b_size,lambda, gLineSearch);
                xpby(xbar, gLineSearch, var1, 1.0/L, d);
                obj_valueKplus = objective_funSample(var1, vals,colidx,pointerB,pointerE, descrA , y , d, b_size, lambda);
                obj_valueK = objective_funSample(xbar, vals,colidx,pointerB,pointerE, descrA , y , d, b_size, lambda);
                obj_valueK = obj_valueK - 1.0/(2*L)*ddot(&d, gLineSearch, &inc, gLineSearch, &inc);

            }

            eta = 1.0/L;
            //printf("%f\n",eta );
            // **********
            for (i = 0; i < p; ++i)
            {
                // op: z= x+delta
                memset(z, 0, d*sizeof(double));
                cblas_daxpy (d, 1, xbar, 1, z, 1);
                cblas_daxpy (d, 1, delta, 1, z, 1);

                // op: g = gradient(z)
                // op: xplus = z - eta * g
                memset(g, 0, d*sizeof(double));
                gradfun(z , vals, colidx, rowidx, descrA, y, d ,m,  m_local,lambda, g);
                xpby(z, g, xplus, eta, d);

                // op: delta = (mlocal/m) * (xplus-x) + delta
                double pi = (1.0 * m_local)/m;
                xpby(xplus, x, xplus_x, 1, d); // op: xplus-x
                cblas_daxpy (d, pi, xplus_x, 1, delta, 1);
                //
                memcpy (x, xplus, d*sizeof(double)); 
            }
            


            // send results(delta) to the master
            MPI_Request request;
            MPI_Isend(
                delta,
                d,
                MPI_DOUBLE,
                0,
                0,
                MPI_COMM_WORLD,
                &request);
            MPI_Request_free(&request);
            //printf("rank %d:  send update to master - waiting to Recv x\n",rank);
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
            //printf("rank %d: Recv x finished \n",rank);
            if(status.MPI_TAG == 3){
                printf("rank %d: Recieved signal. Finishing the process \n", rank);
                break;
            }            
            
        }

        free(delta);
        free(z);
        free(g);
        free(xplus);
        free(xplus_x);
        free(gLineSearch);
        free(pointerB);
        free(pointerE);
        free(var1);
        
    }

    if (rank ==0)
    {   
        double *delta =(double*) malloc (d*sizeof(double));
     
        int it = 0;
        while(it < Iter){
            MPI_Status status;
            MPI_Recv(
            delta,
            d,
            MPI_DOUBLE,
            MPI_ANY_SOURCE,
            0,
            MPI_COMM_WORLD,
            &status);

            // op: x = x + delta
            cblas_daxpy (d, 1, delta, 1, x, 1);

            if(it%freq == 0 ){
                //printf("it/freq +1 %d\n",it/freq +1 );
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
                for (i = 1; i < world_size; ++i)
                {
                    printf("rank %d: ROOT: sent finish signal to rank %d\n", rank, i );
                    MPI_Send(x, d, MPI_DOUBLE, i, 3, MPI_COMM_WORLD);
                }
            }
            

        it++;
        }
        free(delta);

    }


    // Finalize the MPI environment.
    MPI_Finalize();

    // the next 2 lines are for computing the objective function
    // to report later
    
    double *gval =(double*) malloc (d*sizeof(double));
    if(rank == 0){
        for (i = 0; i < (Iter-1)/freq; ++i)
    {
        //double res = 0;
        //printf("%d\n", i);
        double obj_value = objective_fun(cumX+i*d, vals,colidx,rowidx, descrA , y , d, m, lambda);
        gradfun(cumX+i*d , vals,colidx,rowidx, descrA, y, d ,m,  m_local,lambda, gval);
        double grad_value = ddot(&d, gval, &inc, gval, &inc);
        printf(" %ld , %.8f,%.8f \n", times[i], obj_value,grad_value);
    }
    }
    return 0;
}