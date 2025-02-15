import java.util.*;
import java.io.*;

public class test{

    public LinkedHashMap<String, Double> forecastResult;
    public List<Layer> newLayer;
    
    public void tester(List<Layer> lyrs, LinkedHashMap<String, double[]> featureSet, LinkedHashMap<String, double[]> trainSet, LinkedHashMap<String, double[]> testSet,int t_dates, int t_tr_dates, int depNum, File OutFile, File ResultFile){
	    System.out.println("Testing");
	    int f_size = lyrs.get(0).forecast_size;
	    //int depNum=lyrs.get(0).depNum;
	    
	    String[] keys = featureSet.keySet().toArray(new String[0]);
	    
	    double[] inp_val;
	    double[] out_val;
	    double t_error=0;
	    int store_num=1;
	    int i=0;

	    forecastResult = new LinkedHashMap<String, Double>();
	    
	    newLayer = new ArrayList<Layer>();
	    List<Layer> newLayer1 = new ArrayList<Layer>();

	   	    
	    
	    weightsUpload(lyrs, OutFile);

	    System.out.println(newLayer.get(0).input2hidden[0][0]);


	    for(int l_index=(newLayer.size()-1);l_index>=0;l_index--)
		{
		    newLayer1.add(newLayer.get(l_index));
		}

	    newLayer=newLayer1;
	

	    System.out.println("Weights Updated");

	 
	    
	    while (i<keys.length-f_size+1){
			inp_val = new double[lyrs.get(0).input_size];
			out_val = new double[depNum];

			String keyTrain = keys[i+f_size-1]+"b1";
	    
			if(!testSet.containsKey(keyTrain))
			    {
    				i++;
			     	continue;
			    }
			fullBackProp f = new fullBackProp();
		       	inp_val = forecastInput(i, keys, featureSet, trainSet, depNum, f_size);
		        String str1 = keys[i + f_size-1];
		
			double[] temp;
			    
			temp = new double[depNum];
			    
			for (int j=1; j<=depNum;j++)
			    {
				String str2= str1 + "b" + j;
				if(testSet.containsKey(str2))
				    temp[j-1] = testSet.get(str2)[2];
				else 
				    temp[j-1] = 0;
			    }
			   out_val = temp;
			   // System.out.println("Prediction for date : " + keys[i]);
			   
		       double[] newOutput = f.getLayerOutput(newLayer, 3, inp_val);
		       
		       for (int ind = 1; ind <=newOutput.length; ind++){
			   String str2= str1 + "b" + ind;
			   forecastResult.put(str2, newOutput[ind-1]);
		           //System.out.println(-10000 + newOutput[ind]*510000);
		       }
		       i++;
		       // System.out.println("\n\n");
	    }

	    try{
	  
		FileWriter fw = new FileWriter(ResultFile);
       	
		BufferedWriter bw = new BufferedWriter(fw);

		for(String f_keys: forecastResult.keySet())
		    {
			double sales = -10000.0 + forecastResult.get(f_keys)*510000;
			String wr = f_keys + "," + sales;
			bw.write(wr);
			bw.newLine();
		    }
		bw.close();
	    }catch (Exception e){
	    e.printStackTrace();
	    System.exit(-1);
	    }
	

	    

	}


    public double[] forecastInput(int i, String[] All_Keys, LinkedHashMap<String, double[]> featureSet, LinkedHashMap<String, double[]> trainSet, int depNum, int forecast_size )
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

			if(trainSet.containsKey(All_Keys[i1]))
			    {
				
				temp = new double[depNum];
			    
				for (int j=1; j<=depNum;j++)
				    {
		       	
					String keyTrain1 = All_Keys[i+i1]+"b"+j;

					if(!trainSet.containsKey(keyTrain1))
					    temp[j-1]=0;
					else
					    temp[j-1] = trainSet.get(keyTrain1)[2];
					      
				    }
			    }
			else
			    {
		           
				temp = new double[depNum];
			    
				for (int j=1; j<=depNum;j++)
				    {
		       	
					String keyTrain1 = All_Keys[i+i1]+"b"+j;

					if(!forecastResult.containsKey(keyTrain1))
					    temp[j-1]=0;
					else
					    temp[j-1] = forecastResult.get(keyTrain1);
					      
				    }
			    }
		       
						
			 input_val = merge(input_val, temp); 
		    }
	    }	

	return input_val;

    }

    public void  weightsUpload(List<Layer> lyrs, File outFile)
    {

	Layer layer=lyrs.get(0);

	   
	try{
	  
	    FileReader fr = new FileReader(outFile);
       	
	    BufferedReader br = new BufferedReader(fr);		
	
	    String line;

	    int counter=0;
	    

	 
	    int initialFlag=0;
	    

	    while((line=br.readLine())!=null)
		{

		    // System.out.println(line);
		    
		
		    int layer_num;
		    String[] splitLine = line.split(" ");
		    
		    if(splitLine.length==1 && counter==0)
			{
			    if(initialFlag>0)
				{
				    newLayer.add(layer);
				    
				    System.out.println(layer.layer_num + "  added " + layer.input_size); 
				}

			   

			    layer_num=Integer.parseInt(splitLine[0]);
			    layer= lyrs.get(layer_num-1);
			    counter++;
				
			}
		    else if(splitLine.length==1 && counter>0)
			{
			    counter=0;
			    
			}
			
		    if(splitLine.length>1)
			{
			    initialFlag=1;
			    int k=Integer.parseInt(splitLine[0]);
			    int j=Integer.parseInt(splitLine[1]);
			    double weight=Double.parseDouble(splitLine[2]);
	      
			    layer.input2hidden[j][k]=weight; 
			}
		}
	    
	    
	    br.close();
	}catch (Exception e){
	    e.printStackTrace();
	    System.exit(-1);
	}

	 newLayer.add(layer);
	 System.out.println(layer.layer_num + "  added " + layer.input_size);
	System.out.println("Layer 1");
	System.out.println(newLayer.get(1).input_size +  "  " + newLayer.get(1).hidden_size);
	
			
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


}
