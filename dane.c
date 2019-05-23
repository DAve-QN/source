#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mkl.h>
#include <time.h>
#include <sys/time.h>

int cols;
double val, x;
char* line;
int max_line_len = 1024*10;

void printArray(double *arr, int n){
    int i;
    for (i = 0; i < n; ++i)
    {
        printf("%.10f\n",arr[i] );
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



void local_solver(double *x,
    double *vals,
    int *colidx,
    int *rowidx,
    char *descr,
    double *y ,
     int d,
     int m,
     int m_local,
     double *g,
     double *gsum,
     double eta,
     double mio,
     double lambda,
     double *xs,
     int seed,
     double gamma){



    // TODO SEED
    srand(seed);
    // implementing SVRG
    // SVRG parameters:
    int S = 4; // outer iterations
    int M = m_local; // inner iterations
    //double gamma = 0.01; // should be tuned
    int b_size = 1;
    int i,j,s;

    double *w = (double*) malloc (d*sizeof(double));
    double *wbar = (double*) malloc (d*sizeof(double));
    double *gwbar = (double*) malloc (d*sizeof(double));
    int *pointerE = (int*) malloc (b_size*sizeof(int));
    int *pointerB = (int*) malloc (b_size*sizeof(int));
    double *g1 = (double*) malloc (d*sizeof(double));
    double *g2 = (double*) malloc (d*sizeof(double));

    // g = g -eta*gsum
    cblas_daxpby (d, -eta, gsum, 1, 1, g, 1);
            //printArray(gsum,20);

    memcpy (w, x, d*sizeof(double)); 

    for (s = 0; s < S; ++s)
    {
        memcpy (wbar, w, d*sizeof(double)); 
        gradfun(wbar , vals, colidx, rowidx, rowidx+1, descr, y, d ,m, m_local,lambda, gwbar);
    
        
        // gwbar-=g gwbar+=mio*wbar gwbar-=mio*x

        cblas_daxpby (d, -1, g, 1, 1, gwbar, 1);
        cblas_daxpby (d, mio, wbar, 1, 1, gwbar, 1);
        cblas_daxpby (d, -mio, x, 1, 1, gwbar, 1);
        

        for (i = 0; i < M; ++i)
        {
            
            // first generate the random samples
            // by making pointerE and pointerB
            // pointing to the end and the beginning
            // of sampled rows[read MKL manual at:https://software.intel.com/en-us/mkl-developer-reference-c-sparse-blas-csr-matrix-storage-format]
            for (j = 0; j < b_size; ++j)
            {
                int ind = rand()%m_local;
                pointerB[j] = rowidx[ind];
                pointerE[j] = rowidx[ind+1];
            }
            // sparse_matrix_t csrSample;
            // mkl_sparse_d_create_csr ( &csrSample, SPARSE_INDEX_BASE_ONE,
            //                         b_size,  // number of rows
            //                         d,  // number of cols
            //                         pointerB,
            //                         pointerE,
            //                         colidx,
            //                         vals );
            // g1:
            gradfun(w , vals, colidx, pointerB,pointerE, descr, y, d ,m, b_size,lambda, g1);

            // g2:
            gradfun(wbar , vals, colidx, pointerB,pointerE, descr, y, d ,m, b_size,lambda, g2);

            cblas_daxpby (d, -1, g2, 1, 1, g1, 1);

            cblas_daxpby (d, mio, w, 1, 1, g1, 1);
            cblas_daxpby (d, -mio, wbar, 1, 1, g1, 1);
            //printArray(g1,20);
            cblas_daxpby (d, 1, gwbar, 1, 1, g1, 1);
            //printArray(gwbar,20);
            cblas_daxpby (d, -gamma, g1, 1, 1, w, 1);


        }
    }

//printArray(w,20);
    //memset(w,0,d*sizeof(double));
    memcpy (xs, w, d*sizeof(double)); 



    free(w);
    free(wbar);
    free(gwbar);
    free(pointerB);
    free(pointerE);
    free(g1);
    free(g2);


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
    int world_size;
    int rank;
    struct timeval start,end;

    // Get the number of processes and rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // Print off a hello world message
    printf("rank %d: started\n",
     rank);
    if(world_size<1){
        printf("%s\n", "There should be at least one processors!");
        MPI_Finalize();
        return 0;
    }



// INPUT
    if(argc<11){
        printf("Input Format: pathname nrows nnz ncols iterations lambda eta[dane step size: default = 1] mio gamma[svrg step size] [freq of printing]\n");
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
    double mio = atof(argv[8]);
    double gamma = atof(argv[9]);
    int freq = atoi(argv[10]);

    int seed = 42;

    if(rank==0){
        printf("lambda is %f\n",lambda );
        printf("Iter is %d\n",Iter );
        printf("eta is %f\n",eta );
        printf("mio is %f\n",mio );
        printf("gamma is %f\n",gamma );
    }

    // TODO: move to the top
        double one = 1.0;
        MKL_INT inc = 1;
        double zero = 0.0;
        char trans = 'N';
        int cumX_size = (Iter)/freq +20;

    double *cumX = (double *) malloc((cumX_size*d)*sizeof(double));
    long *times = (long*) malloc((cumX_size)*sizeof(long));

    int *rowidx,*colidx, *nnzArray, nnz_local, m_local; // m_local: number of samples for each processor
    double *vals,*y;
    char *descrA;
    int *rowidxFull,*colidxFull, nnz_full, m_full; 
    double *valsFull,*yFull;

    rowidx=(int *)malloc((m+1)*sizeof(int));
    colidx=(int *)malloc(nnz*sizeof(int));
    vals=(double *)malloc(nnz*sizeof(double));
    y=(double *)malloc(m*sizeof(double));
    // just for rank 0
    if(rank == 0){
        rowidxFull=(int *)malloc((m+1)*sizeof(int));
        colidxFull=(int *)malloc(nnz*sizeof(int));
        valsFull=(double *)malloc(nnz*sizeof(double));
        yFull=(double *)malloc(m*sizeof(double));
    }
    

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
    // sparse_matrix_t csrA;
    // sparse_matrix_t csrAFull;

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


    char pathBuff[1000];
    size_t destination_size = sizeof (pathBuff);
    strncpy(pathBuff, pathName, destination_size);
    pathBuff[destination_size - 1] = '\0';

    char buf[100];
    strcat(pathBuff, "-");
    sprintf(buf, "%d", rank);
    strcat(pathBuff,buf);
        
    readgg(pathBuff, rowidx, colidx, vals, y, &nnz_local, &m_local, nnzArray);
        
        // normalizing the matrix between 0 and 1
        // first we get the max and min of local dataset
        // then we reduce it to get the overall max and min
        
    getMaxMin(colidx, vals, d, nnz_local, m_local, max, min,nnzArray);

    
    // get the maximum and minimum over all processors
    MPI_Allreduce(min, minAll, d, MPI_DOUBLE, MPI_MIN,
           MPI_COMM_WORLD);
    MPI_Allreduce(max, maxAll, d, MPI_DOUBLE, MPI_MAX,
           MPI_COMM_WORLD);

    //normalize the dataset using maxall and minall
    // which contain the max and min for each column

    normalize(colidx, vals, d, nnz_local, maxAll, minAll);

    // mkl_sparse_d_create_csr ( &csrA, SPARSE_INDEX_BASE_ONE,
    //                                 m_local,  // number of rows
    //                                 d,  // number of cols
    //                                 rowidx,
    //                                 rowidx+1,
    //                                 colidx,
    //                                 vals );
        // initial gradients 
    gradfun(x , vals, colidx, rowidx, rowidx+1, descrA, y, d ,m, m_local,lambda, g);
    

    if(rank == 0 ){

        // just for objective calculation
        // we don't need this for performance evaluation:
        readgg(pathName, rowidxFull, colidxFull, valsFull, yFull, &nnz_full, &m_full,nnzArray);
        normalize(colidxFull, valsFull, d, nnz_full, maxAll, minAll);
        //printArray(vals,m);
        // Create handle with matrix stored in CSR format
        //printf("JUICY\n");
        // mkl_sparse_d_create_csr ( &csrAFull, SPARSE_INDEX_BASE_ONE,
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
    // could be removed most likely
    double neta= -0.001/(world_size);
    cblas_daxpy (d, neta, gsum, 1, x, 1);
    
    // updated x's 
    double *xs =(double*) malloc (d*sizeof(double));

    //start timing
    if(rank ==0){
        memcpy (cumX, x, d*sizeof(double)); 
        gettimeofday(&start,NULL);
        times[0] = 0;
    }
    // starting the algorithm
    int it = 0;
    while(it < Iter){
        
        // compute local gradients
        gradfun(x , vals, colidx, rowidx, rowidx+1, descrA, y, d ,m, m_local,lambda, g);
        MPI_Allreduce(g, gsum, d, MPI_DOUBLE, MPI_SUM,
           MPI_COMM_WORLD);
        // average the gradeints:
        //printf("%d\n",world_size );
        cblas_daxpby (d, 0, g, 1, 1.0/world_size, gsum, 1);
        local_solver(x, vals,colidx,rowidx, descrA, y, d, m, m_local, g, gsum, eta, mio,lambda, xs, seed, gamma);
        seed++;

        MPI_Allreduce(xs, x, d, MPI_DOUBLE, MPI_SUM,
           MPI_COMM_WORLD);
        // average the x's:
        cblas_daxpby (d, 0, xs, 1, 1.0/world_size, x, 1);
        
        if(it%freq == 0 ){
                //printf("it/freq +1 %d\n",it/freq +1 );
                gettimeofday(&end,NULL);
                long seconds = end.tv_sec - start.tv_sec;
                times[it/freq +1] = (seconds*1000)+(end.tv_usec - start.tv_usec)/1000;
                memcpy (cumX+(it/freq +1)*d, x, d*sizeof(double)); 
        }
        it++;
    }
    free(xs);
    free(x);
    free(min);
    free(max);
    free(g);
    free(gsum);
    free(maxAll);
    free(minAll);


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
        double obj_value = objective_fun(cumX+i*d, valsFull,colidxFull,rowidxFull, descrA , yFull , d, m, lambda);
        gradfun(cumX+i*d , valsFull,colidxFull,rowidxFull, rowidxFull+1, descrA, yFull, d ,m,  m_full,lambda, gval);
        double grad_value = ddot(&d, gval, &inc, gval, &inc);
        printf(" %ld , %.8f,%.8f \n", times[i], obj_value,grad_value);
    }
    }
    return 0;
}