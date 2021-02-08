/*
 * Fichier source pour le projet d'unité
 *  INF-4101C
 *---------------------------------------------------------------------------------------------------
 * Pour compiler : g++ `pkg-config --cflags opencv` projet.cpp `pkg-config --libs opencv` -o projet
 *---------------------------------------------------------------------------------------------------
 * auteur : Eva Dokladalova 09/2015
 * modification : Eva Dokladalova 10/2017
 */


/* 
 * Libraries stantards 
 *
 */ 
#include <stdio.h>
#include <stdlib.h>

/* 
 * Libraries OpenCV "obligatoires" 
 *
 */ 
#include "highgui.h"
#include "cv.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
  
/* 
 * Définition des "namespace" pour évite cv::, std::, ou autre
 *
 */  
using namespace std;
using namespace cv;
using std::cout;

/*
 * Some usefull defines
 * (comment if not used)
 */
#define PROFILE
#define VAR_KERNEL
#define N_ITER 100

#ifdef PROFILE
#include <time.h>
#include <sys/time.h>
#include <assert.h>     /* assert */
#endif

/**
 * This structure represents a two-tier histogram. The first tier (known as the
 * "coarse" level) is 4 bit wide and the second tier (known as the "fine" level)
 * is 8 bit wide. Pixels inserted in the fine level also get inserted into the
 * coarse bucket designated by the 4 MSBs of the fine bucket value.
 *
 * The structure is aligned on 16 bits, which is a prerequisite for SIMD
 * instructions. Each bucket is 16 bit wide, which means that extra care must be
 * taken to prevent overflow.
 */
typedef struct
{
    unsigned short coarse[16];
    unsigned short fine[16][16];
} Histogram;

void afficheHisto(Histogram h)
{
	int i;
	int j;
	
	for(i=0;i<16;i++)
	{
		printf("Coarse : %d = ",h.coarse[i]);
		for(j=0;j<16;j++)
		{
			printf("%d +",h.fine[i][j]);
		}
		printf("\n");
	}
}

uchar histoMedian(Histogram histo, int r)
{
	int temp = 0;
	int comp = (((2*r)+1)*((2*r)+1))/2;
	int res, i, k;
	for(i=0;i<16; i++)
	{
		temp+=histo.coarse[i];
		if (temp>comp)
		{
			temp-=histo.coarse[i];
			break;
		}
	}
	assert(i<16);
	
	for(k=0;k<16;k++)
	{
		temp+=histo.fine[i][k];
		if(temp>comp)
		{
			res=(uchar) i*16+k;
			break;
		}
	}
	assert(k<16);
	return res;
}

#if CV_SSE2
#define MEDIAN_HAVE_SIMD 1

static inline void histogram_add_simd( const unsigned short x[16], unsigned short y[16] )
{
    const __m128i* rx = (const __m128i*)x;
    __m128i* ry = (__m128i*)y;
    __m128i r0 = _mm_add_epi16(_mm_load_si128(ry+0),_mm_load_si128(rx+0));
    __m128i r1 = _mm_add_epi16(_mm_load_si128(ry+1),_mm_load_si128(rx+1));
    _mm_store_si128(ry+0, r0);
    _mm_store_si128(ry+1, r1);
}

static inline void histogram_sub_simd( const unsigned short x[16], unsigned short y[16] )
{
    const __m128i* rx = (const __m128i*)x;
    __m128i* ry = (__m128i*)y;
    __m128i r0 = _mm_sub_epi16(_mm_load_si128(ry+0),_mm_load_si128(rx+0));
    __m128i r1 = _mm_sub_epi16(_mm_load_si128(ry+1),_mm_load_si128(rx+1));
    _mm_store_si128(ry+0, r0);
    _mm_store_si128(ry+1, r1);
}

#else
#define MEDIAN_HAVE_SIMD 0
#endif

static inline void histogram_add( const unsigned short x[16], unsigned short y[16] )
{
    int i;
    for( i = 0; i < 16; ++i )
        y[i] = (unsigned short)(y[i] + x[i]);
}

static inline void histogram_sub( const unsigned short x[16], unsigned short y[16] )
{
    int i;
    for( i = 0; i < 16; ++i )
        y[i] = (unsigned short)(y[i] - x[i]);
}

void histoOp(Histogram &histo, Histogram* column_histo, int index, int sign)
{
#if MEDIAN_HAVE_SIMD
    volatile bool useSIMD = checkHardwareSupport(CV_CPU_SSE2);
#endif

	int j;
	
	#if MEDIAN_HAVE_SIMD
		if( useSIMD )
		{
			//update le coarse
			if (sign==1) histogram_add_simd(column_histo[index].coarse, histo.coarse);
			else histogram_sub_simd(column_histo[index].coarse, histo.coarse);

			for(j=0;j<16;j++)
			{
				//update le fine si le coarse est != de 0
				//tester avec et sans le if pour 259
				if(column_histo[index].coarse[j]!=0)
				{
					if (sign==1) histogram_add_simd(column_histo[index].fine[j], histo.fine[j]);
					else histogram_sub_simd(column_histo[index].fine[j], histo.fine[j]);
				}
			}
		}
		else
	#endif
		{
			//update le coarse
			if (sign==1) histogram_add(column_histo[index].coarse, histo.coarse);
			else histogram_sub(column_histo[index].coarse, histo.coarse);

			for(j=0;j<16;j++)
			{
				//update le fine si le coarse est != de 0
				if(column_histo[index].coarse[j]!=0)
				{
					if (sign==1) histogram_add(column_histo[index].fine[j], histo.fine[j]);
					else histogram_sub(column_histo[index].fine[j], histo.fine[j]);
				}
			}
		}
}

bool condition(int i, int j, int borne)
{
	if (i%2==0) return j<borne;
	return j>=borne;
}

void median(Mat& frame1, Mat& frame2, int kernel)
{
	frame2 = Mat(frame1.rows-kernel+1, frame1.cols-kernel+1, frame1.type());
	int r = kernel/2;
	int i, j, k, start, y, borne, it, y2;
	uchar px1, px2;
	Histogram histo;
	Histogram* column_histo;
	column_histo = (Histogram*) malloc(frame1.cols*sizeof(Histogram));
	
	// set all values to 0
	for(i=0;i<16;i++)
	{
		histo.coarse[i]=0;
		for(j=0;j<16;j++)
		{
			histo.fine[i][j]=0;
		}
	}
	
	for(k=0;k<frame1.cols;k++)
	{
		for(i=0;i<16;i++)
		{
			column_histo[k].coarse[i]=0;
			for(j=0;j<16;j++)
			{
				column_histo[k].fine[i][j]=0;
			}
		}
	}
	
	//initialisation
	//histogramme colonne
	for(i = 0; i < kernel; i++)
	{		
		for(j = 0; j < frame1.cols; j++)
		{
			px1 = frame1.at<uchar>(i,j);
			
			column_histo[j].coarse[px1/16]+=1;
			column_histo[j].fine[px1/16][px1%16]+=1;
		}
	}
	
	//premier point
	for(i = 0; i < kernel; i++)
	{
		//histo+=column_histo de 0 à kernel-1
		histoOp(histo, column_histo, i, 1);
	}
	frame2.at<uchar>(0,0) = histoMedian(histo, r);
	
	//calcul de la médiane de la première ligne fait à part car
	//on ne peut pas retirer le pixel du haut. Plus rapide de le faire à part que de faire un test
	//pour chaque ligne
	for(i=1;i<frame2.cols;i++)
	{
		histoOp(histo, column_histo, kernel+i-1, 1);
		histoOp(histo, column_histo, i-1, -1);
		frame2.at<uchar>(0,i) = histoMedian(histo, r);
	}
	
	//passage de la premiere ligne
	//calcul en O(r)
	for(i=0;i<kernel;i++)
	{
		px1=frame1.at<uchar>(0,(frame2.cols-1+i));
		px2=frame1.at<uchar>(kernel,(frame2.cols-1+i));
		
		column_histo[(frame2.cols-1+i)].coarse[px1/16]-=1;
		column_histo[(frame2.cols-1+i)].fine[px1/16][px1%16]-=1;
		
		column_histo[(frame2.cols-1+i)].coarse[px2/16]+=1;
		column_histo[(frame2.cols-1+i)].fine[px2/16][px2%16]+=1;
		
		histo.coarse[px2/16]+=1;
		histo.fine[px2/16][px2%16]+=1;
		
		histo.coarse[px1/16]-=1;
		histo.fine[px1/16][px1%16]-=1;
	}
	frame2.at<uchar>(1,frame2.cols-1) = histoMedian(histo, r);
	
	for(i=1;i<frame2.rows;i++)
	{
		if(i%2==0)
		{
			start = 1;
			borne = frame2.cols;
			it=1;
			y = -1+kernel;
			y2 = -1;
		}
		else
		{
			start = frame1.cols-kernel-1;
			borne = 0;
			it=-1;
			y = 0;
			y2 = 0;
		}
		
		//update tous les histo de la ligne
		for(j=start;condition(i, j, borne);j+=it)
		{
			px1=frame1.at<uchar>(i+kernel-1,j+y);
			px2=frame1.at<uchar>(i-1,j+y);
			
			column_histo[(j+y)].coarse[px1/16]+=1;
			column_histo[(j+y)].fine[px1/16][px1%16]+=1;			
			
			column_histo[(j+y)].coarse[px2/16]-=1;
			column_histo[(j+y)].fine[px2/16][px2%16]-=1;
		}
		
		//calculer les médianes de la ligne en additionnant et supprimant les colonnes d'histo
		for(j=start;condition(i, j, borne);j+=it)
		{
			histoOp(histo, column_histo, j+kernel+y2, it);
			histoOp(histo, column_histo, j+y2, -it);

			frame2.at<uchar>(i,j) = histoMedian(histo, r);
		}
		j-=it;

		//PASSER LA LIGNE
		if(i!=frame2.rows)
		{
			for(k=0;k<kernel;k++)
			{
				px1=frame1.at<uchar>(i,j+k);
				px2=frame1.at<uchar>(i+kernel,j+k);

				column_histo[j+k].coarse[px1/16]-=1;
				column_histo[j+k].fine[px1/16][px1%16]-=1;

				column_histo[j+k].coarse[px2/16]+=1;
				column_histo[j+k].fine[px2/16][px2%16]+=1;

				histo.coarse[px2/16]+=1;
				histo.fine[px2/16][px2%16]+=1;

				histo.coarse[px1/16]-=1;
				histo.fine[px1/16][px1%16]-=1;
			}
			frame2.at<uchar>(i+1,j) = histoMedian(histo, r);
		}
	}
	
	free(column_histo);
}

/*
 *
 *--------------- MAIN FUNCTION ---------------
 *
 */
int main () {
//----------------------------------------------
// Video acquisition - opening
//----------------------------------------------
  VideoCapture cap(0); // le numéro 0 indique le point d'accès à la caméra 0 => /dev/video0
  if(!cap.isOpened()){
    cout << "Errore"; return -1;
  }

//----------------------------------------------
// Déclaration des variables - imagesize
// Mat - structure contenant l'image 2D niveau de gris
// Mat3b - structure contenant l'image 2D en couleur (trois cannaux)
//
  Mat3b frame; // couleur
  Mat frame1; // niveau de gris 
  Mat frame_gray; // niveau de gris 
  Mat grad_x;
  Mat grad_y;
  Mat abs_grad_y;
  Mat abs_grad_x;
  Mat grad;

// variable contenant les paramètres des images ou d'éxécution  
  int ddepth = CV_16S;
  int scale = 1;
  int delta = 0;	
  unsigned char key = '0';

 #define PROFILE
  
#ifdef PROFILE
// profiling / instrumentation libraries
#include <time.h>
#include <sys/time.h>
#endif
  
//----------------------------------------------------
// Création des fenêtres pour affichage des résultats
// vous pouvez ne pas les utiliser ou ajouter selon ces exemple
// 
  cvNamedWindow("Video input", WINDOW_AUTOSIZE);
  cvNamedWindow("Video gray levels", WINDOW_AUTOSIZE);
  cvNamedWindow("Video Mediane", WINDOW_AUTOSIZE);
  cvNamedWindow("Video Edge detection", WINDOW_AUTOSIZE);
// placement arbitraire des  fenêtre sur écran 
// sinon les fenêtres sont superposée l'une sur l'autre
  cvMoveWindow("Video input", 10, 30);
  cvMoveWindow("Video gray levels", 800, 30);
  cvMoveWindow("Video Mediane", 10, 500);
  cvMoveWindow("Video Edge detection", 800, 500);
  
  
// --------------------------------------------------
// boucle infinie pour traiter la séquence vidéo  
//
  while(key!='q'){
  //	  
  // acquisition d'une trame video - librairie OpenCV
    cap.read(frame);
  //conversion en niveau de gris - librairie OpenCV
    cvtColor(frame, frame_gray, CV_BGR2GRAY);

	
   // image smoothing by median blur
   // 
 int n = 259;
 int k = 1;  
 #ifdef PROFILE
 struct timeval start, end;
 for (k;k<n;k+=2)
{ 
 gettimeofday(&start, NULL);
 #endif
     median(frame_gray, frame1, k);
 #ifdef PROFILE
 gettimeofday(&end, NULL);
 double e = ((double) end.tv_sec * 1000000.0 + (double) end.tv_usec);
 double s = ((double) start.tv_sec * 1000000.0 + (double) start.tv_usec);
 printf("%d\n", int(e - s));
}
	return 0;
 #endif
    
	// ------------------------------------------------
	// calcul du gradient- librairie OpenCV
    /// Gradient Y
    Sobel( frame1, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	/// absolute value
    convertScaleAbs( grad_x, abs_grad_x );
    /// Gradient Y
    Sobel( frame1, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	/// absolute value
    convertScaleAbs( grad_y, abs_grad_y );
    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad ); 	
    
    // -------------------------------------------------
	// visualisation
	// taille d'image réduite pour meuilleure disposition sur écran
    //    resize(frame, frame, Size(), 0.5, 0.5);
    //    resize(frame_gray, frame_gray, Size(), 0.5, 0.5);
    //    resize(grad, grad, Size(), 0.5, 0.5);
    imshow("Video input",frame);
    imshow("Video gray levels",frame_gray);
    imshow("Video Mediane",frame1);    
    imshow("Video Edge detection",grad);  
    
    
    key=waitKey(5);
  }
}

    
