import java.io.*;
import java.util.*;


public class fullBackProp{

    public void Train(List<Layer> lyrs, LinkedHashMap<String, double[]> featureSet, LinkedHashMap<String, double[]> trainSet, int t_dates, int printIterationGap, File weightFile)// final finetuning step
        {
	    System.out.println("Training");
            int f_size = lyrs.get(0).forecast_size;
	    int depNum=lyrs.get(0).depNum;
	    
	    String[] keys = featureSet.keySet().toArray(new String[0]);
	    
	    double[] inp_val;
	    double[] out_val;
	    boolean printFlag;
	    for(int iter =0; iter<10000; iter++)
		{
		    double t_error=0;
		    printFlag = false;
		    System.out.println("Full Back Prop " + iter);

		    int store_num=1;
		    int i=0;
        	
		    while (i<keys.length-f_size+1)
		    {
			inp_val = new double[lyrs.get(0).input_size];
			out_val = new double[depNum];
			
			

			String keyTrain = keys[i+f_size-1]+"b1";
		       
			
			if(!trainSet.containsKey(keyTrain))
			    {
				i=store_num*t_dates;
				store_num++;
			     	continue;
			    }
		       	inp_val= forecastInput(i, keys, featureSet, trainSet, depNum, f_size);
		        String str1 = keys[i + f_size-1];
			   
		
		        double[] temp;
			    
			temp = new double[depNum];
			    
			for (int j=1; j<=depNum;j++)
			    {
				String str2= str1 + "b" + j;
				if(trainSet.containsKey(str2))
				    temp[j-1] = trainSet.get(str2)[2];
				else 
				    temp[j-1] = 0;
	  
			    }
			    
			   out_val = temp;
			
			   if ((iter%printIterationGap) == 0) printFlag = true;
			   
			   else printFlag = false;
			   
		       finalBackProp(lyrs, inp_val, out_val, printFlag, weightFile, iter);
		       
		       i++;

		       double[] newOutput = getLayerOutput(lyrs, 3, inp_val);
    			for(int er=0; er<newOutput.length;er++)
    			    t_error += Math.abs(out_val[er]-newOutput[er]);
		    }

	if (printFlag){
			for (int lyrIndex=lyrs.size()-1; lyrIndex>=0;lyrIndex--)
			    {
				Layer newLayer = lyrs.get(lyrIndex);
				newLayer.printWeights(weightFile, iter);
			    } 
		    }
		    System.out.println("Error = " + t_error);
		}
	}


    public double[] forecastInput(int i, String[] All_Keys, LinkedHashMap<String, double[]> featureSet, LinkedHashMap<String, double[]> trainSet, int depNum, int forecast_size)
    {
	double[] input_val={0.1,0.1};
	
	for (int i1=0; i1<forecast_size; i1++)		       
	    {	   
    		if(i1==0)
		    input_val = featureSet.get(All_Keys[i]);
       	      	else
		    input_val = merge(input_val, featureSet.get(All_Keys[i+i1]));
				
	     	if(i1<forecast_size-1)
		    {
			double[] temp;
		           
		       	temp = new double[depNum];
			    
		      	for (int j=1; j<=depNum;j++)
			    {
		       	
				String keyTrain1 = All_Keys[i+i1]+"b"+j;

				if(!trainSet.containsKey(keyTrain1))
				    temp[j-1]=0;
			       	else
			       	    temp[j-1] = trainSet.get(keyTrain1)[2];
					      
			     }
		       
						
			 input_val = merge(input_val, temp); 
		    }
	    }	

	return input_val;

    }



    public double[] merge(double[] inp_val1, double[] inp_val2)
    {
	int con_len = inp_val1.length + inp_val2.length;
	double[] concat_inp = new double[con_len];
	
	for (int i=0; i<con_len;i++)
	    {
		if(i<inp_val1.length)
		    concat_inp[i]=inp_val1[i];
		else
		    concat_inp[i]=inp_val2[i-inp_val1.length];
	     }
	return concat_inp;
    }
    
    	    
    public double[] getLayerOutput(List<Layer> layers, int layerNum, double[] input_val )
    {
        if (layerNum == 0)
	    return input_val;

	double[] output = new double[layers.get(layerNum-1).hidden_size];
	double[] new_input = new double[layers.get(layerNum-1).input_size];

	new_input = getLayerOutput(layers, layerNum -1, input_val);
        
      
	
	for (int j=0; j<output.length;j++)
	    {
		output[j]=0;
		for (int k=0; k<layers.get(layerNum-1).input_size; k++)
		    output[j] += layers.get(layerNum-1).input2hidden[k][j]*new_input[k];
        
		output[j]=sigmoid(output[j]);
      	    }
	return output;
    }

    public void getError(List<Layer> layers,int layerNum, double[] input_val)
    {
        Layer ThisLayer = layers.get(layerNum -1);
        Layer NextLayer = layers.get(layerNum);
        double[] error = ThisLayer.l_error;
        double[] outputThisLayer = getLayerOutput(layers, layerNum, input_val);

        for (int i = 0 ; i < ThisLayer.hidden_size; i++)
	    {
		error[i] = 0.0;
		for( int j = 0 ; j < NextLayer.hidden_size; j++){
		    error[i] += NextLayer.l_error[j]*NextLayer.input2hidden[i][j]*outputThisLayer[i]*(1-outputThisLayer[i]);
            }
        }
       
	ThisLayer.l_error=error;
    }

    public void getError(List<Layer> layers,int layerNum, double[] finalOutput, double[] input_val)
    {
        Layer ThisLayer = layers.get(layerNum -1);

        double[] error = ThisLayer.l_error;
        double[] output = getLayerOutput(layers, layerNum,  input_val );
        
	for (int i = 0 ; i < ThisLayer.hidden_size; i++)
            error[i] = (-output[i] + finalOutput[i])*(output[i])*(1-output[i]);
        
        ThisLayer.l_error=error;
    }


    public void finalBackProp(List<Layer> layers, double[] input_val, double[] actualOutput, boolean printFlag, File filename, int iterationNum)
    {
        double rate = layers.get(2).learnrate;

        
        for (int i = (layers.size() - 1); i >= 0; i--)
	    {
		Layer ThisLayer  = layers.get(i);
		
		double[] prevOutput;

		if (i >= 1) 
		    prevOutput = getLayerOutput(layers, ThisLayer.layer_num-1, input_val);
		else
		    prevOutput = input_val;

		double[] Output = getLayerOutput(layers, ThisLayer.layer_num, input_val);

		if (ThisLayer.isFinal)
		    getError(layers, ThisLayer.layer_num, actualOutput, input_val);
        
		else
		    getError(layers, ThisLayer.layer_num, input_val);
                
		for (int k = 0 ; k < ThisLayer.hidden_size; k++)
			for(int j = 0; j < ThisLayer.input_size; j++)
				ThisLayer.input2hidden[j][k] += rate*ThisLayer.l_error[k]*prevOutput[j];
        
                }
        
     }

    public double sigmoid(double t)
    {
	return (1/(1+Math.exp(-t)));
    }
     
    
}
