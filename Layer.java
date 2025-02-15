import java.util.*;
import java.io.*;


class Layer{

    public int input_size;
    public int output_size;
    public int hidden_size;
    public int forecast_size;
    public int layer_num;
    public double[] input_val;
    public double[] output_val;
    public double[] hidden_val;
    public double[][] input2hidden;
    public double[][] hidden2output;
    public double learnrate;
    public boolean isFinal;
    public double[] l_error;
    public double[] delta;
    public int depNum;
    public LinkedHashMap<String, double[]> H_output;

    public Layer(int inp_size, int out_size, int hid_size, int forecast_s, int layer_n, double learn, int d_num)
    {
	
	this.input_size=inp_size;
	
	System.out.println("............." + this.input_size);
      
	this.hidden_size=hid_size;
	this.output_size=out_size;
	this.layer_num=layer_n;
	this.learnrate=learn;
	this.forecast_size=forecast_s;
	this.depNum=d_num;

	if(layer_n==1)
	    {
		this.input_size=input_size*forecast_size + (forecast_size-1)*depNum;
		this.output_size=output_size*forecast_size + (forecast_size-1)*depNum;
	    }
	
	this.input_val = new double[input_size];
	this.hidden_val = new double[hidden_size];
	this.input2hidden = new double[input_size][hidden_size];
	
	for (int i=0; i<input_size; i++)
	    {
	       	this.input_val[i]=0;
       		for (int j=0; j<hidden_size;j++)
		    this.input2hidden[i][j]=(-0.5 + Math.random()*1);
	    }


	if(this.output_size>0)
	    {
		this.delta = new double[output_size];
		this.output_val = new double[output_size];
		this.hidden2output = new double[hidden_size][output_size];
		
		for (int i=0; i<output_size; i++)
		    {
			this.delta[i]=0;
			this.output_val[i]=0;
			for (int j=0; j<hidden_size;j++)
			    {
				this.hidden2output[j][i]=(-0.5 + Math.random()*1) ;
			    }
		    }
	    
		
		this.isFinal=false;
	    }
	else
	    this.isFinal=true;
       
	
	this.l_error = new double[hidden_size];
	for (int j=0; j<this.hidden_size;j++)
	    {
	      	this.hidden_val[j]=0;
		this.l_error[j]=0;
	    }

	System.out.println("Here   " + input_size);

       

	
    }
    
    public LinkedHashMap<String, double[]> getLayerOutput(LinkedHashMap<String, double[]> featureSet, LinkedHashMap<String, double[]> trainSet, int t_dates)
    {
	
	H_output = new LinkedHashMap<String, double[]>();
	    
        Set<String> keys = featureSet.keySet();
     	String[] All_Keys = keys.toArray(new String[0]);
	
        int i=0;
	int store_num=1;
	
	System.out.println(input_size);

        while ( i<All_Keys.length-forecast_size+1)
	       {

		   if(layer_num==1)
			    {
				String keyTrain = All_Keys[i+forecast_size-1]+"b1";
		       
			
				if(!trainSet.containsKey(keyTrain))
				    {
					i=store_num*t_dates;
					store_num++;
					continue;
				    }
				input_val= forecastInput(i, All_Keys, featureSet, trainSet);
			    }

			else
		      	    {
				input_val= featureSet.get(All_Keys[i]);
	       		    }
			
			
		   hidden_val= new double[hidden_size];
	
		   for (int j=0; j<hidden_size;j++)
		       {
       
			   hidden_val[j]=0;
				
			   for (int k=0; k<input_size; k++)
			       hidden_val[j] += input_val[k]*input2hidden[k][j];
			   hidden_val[j]=sigmoid(hidden_val[j]);
				
		       
			       
		       }
		      
		   H_output.put(All_Keys[i], hidden_val);

		   i++;
	       }

	System.out.println(input_size);

	return H_output;

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


    public double[] forecastInput(int i, String[] All_Keys, LinkedHashMap<String, double[]> featureSet, LinkedHashMap<String, double[]> trainSet)
    {
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




    
    public void deepNetwork(LinkedHashMap<String, double[]> featureSet, LinkedHashMap<String, double[]> trainSet,  int t_dates, File file)
    {	
	double t_error=0;
       
   
	Set<String> keys = featureSet.keySet();
	
	String[] All_Keys = keys.toArray(new String[0]);
	int cnt = 0;
	for (int iter =0; iter<5000; iter++)
	    {
		t_error=0;
		int count=0;
		cnt++;
		int store_num=1;
		int i=0;
		
	     
	       
		while (i<All_Keys.length-forecast_size+1)
		    {
		 
			
			if(layer_num==1)
			    {
				String keyTrain = All_Keys[i+forecast_size-1]+"b1";
		       
			
				if(!trainSet.containsKey(keyTrain))
				    {
					i=store_num*t_dates;
					store_num++;
					    continue;
				    }
				input_val= forecastInput(i, All_Keys, featureSet, trainSet);
			    }

			else
		      	    {
				input_val= featureSet.get(All_Keys[i]);
	       		    }

		      

				
			for (int j=0; j<hidden_size;j++)
			    {
				hidden_val[j]=0;
				
				for (int k=0; k<input_size; k++)
				    {
					hidden_val[j] += input_val[k]*input2hidden[k][j];
				     
				    }
				hidden_val[j]=sigmoid(hidden_val[j]);
			      
			    }
			

			for (int j=0; j<output_size;j++)
			    {
				output_val[j]=0;
				
				for (int k=0; k<hidden_size; k++)
				    output_val[j] += hidden_val[k]*hidden2output[k][j];		
				output_val[j]=sigmoid(output_val[j]);

		  
				t_error+=1.0*Math.abs(output_val[j]-input_val[j]);
			    }
		       
			backProp(0);
			backProp(1);
			i++;
			    
		    }
		System.out.println("Error " + t_error);
	    }
	    
	    this.printWeights( file, cnt);
		
    }
    
    public void backProp(int flag)
    {
	double error=0;
	double delta_h=0;
	
	if(flag==0)
	    {
		for (int i=0; i<hidden_size;i++)
		    {
       	    	
       			for (int j=0; j<output_size; j++)
			    {
				error = input_val[j] - output_val[j];
			       
				delta[j] = output_val[j]*(1-output_val[j])*error;
				hidden2output[i][j] += learnrate*delta[j]*hidden_val[i];
			    }
		    }
			    
	    }

	if(flag==1)
	    {
		for (int i=0; i<input_size;i++)
		    {
       	    	
       			for (int j=0; j<hidden_size; j++)
			    {
				delta_h=0.0;
				
				for (int k=0; k<output_size; k++)
				    {
					delta_h += delta[k]*hidden2output[j][k];
					input2hidden[i][j] += learnrate*delta_h*hidden_val[j]*(1-hidden_val[j])*input_val[i];
				    }
				
			    }
		    }
			    
	    }
	
	
	
    }

    public double sigmoid(double t)
    {
	return (1.0/(1+Math.exp(-t)));
    }
    
    public void printWeights(File file, int iternum){
	try{
	    boolean appendFlag = true;// 
	   // if (layer_num > 1 ) appendFlag = true;// creates problem when writing the weights of the final layer (the 1st one) in the fullBackProp
        FileWriter fw = new FileWriter(file.getAbsoluteFile(), appendFlag);
		
		BufferedWriter bw = new BufferedWriter(fw);
		
		bw.write(Integer.toString(layer_num) + "\n");
		bw.write(Integer.toString(iternum));
		
		bw.newLine();
        for (int k = 0 ; k < hidden_size; k++){
			for(int j = 0; j < input_size; j++){
				String weight = Integer.toString(k) + " " + Integer.toString(j) +" " + Double.toString(input2hidden[j][k]); 
			    bw.write(weight);
			    bw.newLine();
			}
        }
        bw.close();
	}catch (Exception e){
	    e.printStackTrace();
	    System.exit(-1);
	}
    }
    
    //  public List<Layer> readLayers(file){
        
        
    // }
    

} 
