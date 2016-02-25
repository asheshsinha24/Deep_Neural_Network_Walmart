import java.io.*;
import java.util.*;

public class DataParser {
	
    public static LinkedHashMap <String, double[]> Feature;
    public static int att_num;
    public static double[] max_val;
    public static double[] min_val;
    public static int hidden_num;
    public static int hidden_num1;
    public static LinkedHashMap <String, double[]> Train;
    public static LinkedHashMap <String, double[]> Test;
    public static int att_num1;
    public static int forecast_size; 
    public static int depNum;
    public static int total_train_dates;
    public static int total_test_dates;
    public static int total_feature_dates;
    public static double[] max_val1;
    public static double[] min_val1;
    public static LinkedHashMap<String, double[]> Hidden_out;
    public static Layer l1,l2,l3;


	public static void main(String[] args) {

	    String featurefile = args[0];
	    String trainfile = args[1];
	    String testFile = args[2];
	    double learningRate = Double.parseDouble(args[3]);
	    double learningRate1 = Double.parseDouble(args[4]);     
	    forecast_size=Integer.parseInt(args[5]);

	    String file1 = "FullBackPropWeight.txt";
	    String file2 = "AutoEncoding.txt";
	    String file3 = "result.txt";
	    String file4 = "FinalWeights.txt";
	    File backPropFile = new File(file1);
	    File AutoEncodeFile = new File(file2);
	    File resultFile = new File(file3);
	    File WeightFile = new File(file4);
	    // if file doesnt exists, then create it
	    if (!backPropFile.exists()) {
		try{		
		backPropFile.createNewFile();
		} catch(Exception e){
		    	e.printStackTrace();
			System.exit(-1);
		}
	    }
	    else{
		try{
		    PrintWriter writer = new PrintWriter(backPropFile);
		    writer.print("");
		    writer.close();
		}catch(Exception e){
		    e.printStackTrace();
		    System.exit(-1);
		}

	    }
    	    if (!AutoEncodeFile.exists()) {
		try{		
		AutoEncodeFile.createNewFile();
		} catch(Exception e){
		    	e.printStackTrace();
			System.exit(-1);
		}
    	    }
	    else{
		try{
		    PrintWriter writer = new PrintWriter(AutoEncodeFile);
		    writer.print("");
		    writer.close();
		} catch(Exception e){
		    System.out.println("File Error");
		    System.exit(-1);
		}
	    }
	    if (!resultFile.exists()) {
		try{		
		resultFile.createNewFile();
		} catch(Exception e){
		    	e.printStackTrace();
			System.exit(-1);
		}
	    }
	    else{
		try{
		    PrintWriter writer = new PrintWriter(resultFile);
		    writer.print("");
		    writer.close();
		} catch(Exception e){
		    System.out.println("File Error");
		    System.exit(-1);
		}
	    }
    	    
	    
	    
	    createDataSet(featurefile);
	    createTrain(trainfile);
	   
	    hidden_num= 40; 
	    hidden_num1= 40;
	    

	    List<Layer> llist = new ArrayList<Layer>();
	    
	    System.out.println("First Layer");
	    l1 = new Layer(att_num-1, att_num-1, hidden_num, forecast_size, 1, learningRate,depNum);
	    
	    System.out.println(l1.layer_num + "  " +  l1.input_size +  "  " + l1.hidden_size);
     
	    //l1.deepNetwork(Feature, Train, total_feature_dates, AutoEncodeFile);
	    //Hidden_out = l1.getLayerOutput(Feature, Train, total_feature_dates);
	    llist.add(l1);
	    
	    System.out.println(l1.layer_num + " " +  l1.input_size +  "  " + l1.hidden_size);
		      
	    System.out.println("Second Layer");
	    l2 = new Layer(hidden_num, hidden_num, hidden_num1,1,2,learningRate, depNum);
	    //l2.deepNetwork(Hidden_out, Train, total_feature_dates, AutoEncodeFile);
	    //LinkedHashMap<String, double[]> finalOut = l2.getLayerOutput(Hidden_out, Train, total_feature_dates);
	    llist.add(l2);
	    
	    System.out.println(l2.layer_num + " " +  l2.input_size +  "  " + l2.hidden_size);
	
	    System.out.println("Final Layer");
	    l3 = new Layer(hidden_num1,0,depNum,1,3,learningRate1, depNum);
	    llist.add(l3);

	    System.out.println(l3.layer_num + " " +  l3.input_size +  "  " + l3.hidden_size);
	    //fullBackProp f1 = new fullBackProp();
	    //f1.Train(llist, Feature, Train, total_feature_dates, 100, backPropFile);
	    
	    createTest(testFile);
	    test t = new test();
	    t.tester(llist, Feature, Train, Test,total_feature_dates, total_train_dates, depNum,  WeightFile, resultFile);

		  
	}


       
	public static void createDataSet(String file) {
      
		BufferedReader in;
		final String DELIMITER = ",";
		
		String dates="a";
		
		total_feature_dates=0;
		
		Feature = new LinkedHashMap<String,double[]>();
		try {
		    
		    att_num=0;

		    in = new BufferedReader(new FileReader(file));

		    while (true)
			{
			    
			    String line = in.readLine(); // %% class Label
			    
			    //System.out.println(line);
			    
			    if(line==null)
				break;
			    String[] splitline = line.split(DELIMITER);

			    if(!dates.equals(splitline[1]) && splitline[0].equals("1"))
			       {
				   dates=splitline[1];
				   total_feature_dates++;
			       }
			    

			    if(att_num==0)
				{
				    att_num=splitline.length;
				    max_val = new double[att_num-1];
				    min_val = new double[att_num-1];
				    
				    for (int i=0; i<att_num; i++)
					{
					    if(i<1)
						{
						    max_val[i]=0;
						    min_val[i]=10000000;
						}
					    else if(i>1)
						{
						    max_val[i-1]=0;
						    min_val[i-1]=10000000;
						}
					}
				}
			    else
				{
				    double f_val[] = new double[att_num-1];
				    
				    String key = splitline[0] + "a"+ splitline[1];
				    
				    for (int i=0; i<att_num;i++)
					{
					    if(i<1)
						{
						    if(splitline[i].equals("NA") | splitline[i].equals("FALSE"))
						       f_val[i]= 0;
						    else if(splitline[i].equals("TRUE"))
						       f_val[i]= 1;
						    else
							f_val[i]=Double.parseDouble(splitline[i]);

						    if(max_val[i]<f_val[i])
							max_val[i]=f_val[i];
						    if(min_val[i]>f_val[i])
							min_val[i]=f_val[i];

						    
						}
					    else if(i>1)
						{
						    if(splitline[i].equals("NA") | splitline[i].equals("FALSE"))
						       f_val[i-1]= 0;
						    else if(splitline[i].equals("TRUE"))
						       f_val[i-1]= 1;
						    else
							f_val[i-1]=Double.parseDouble(splitline[i]);
						    if(max_val[i-1]<f_val[i-1])
							max_val[i-1]=f_val[i-1];
						    if(min_val[i-1]>f_val[i-1])
							min_val[i-1]=f_val[i-1];

						}
					}
						    
						       
					       
				    Feature.put(key, f_val);
				}

			    
			}
			
			    in.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		} 

		
		System.out.println("Feature Data Updated");
	      
		
		for(String key: Feature.keySet())
		    {
			
			double[] f_val = Feature.get(key);
		       
			for (int i=0; i<att_num-1; i++)
			    {
				if(Math.abs(max_val[i]-min_val[i])>0.0001)
				    f_val[i]=0 + 1.0*(f_val[i]-min_val[i])/(max_val[i]-min_val[i]);
				else
				    f_val[i]=1.0;
				
			    }
			Feature.put(key,f_val);
		    }

		for (int i=0; i<att_num-1;i++)
		    System.out.println(max_val[i] + "  " + min_val[i]);
		
	      
	}

    public static void createTrain(String file) {
      
		BufferedReader in;
		final String DELIMITER = ",";
		
		String dates="a";
		
		total_train_dates=0;
		depNum=0;
		
		Train = new LinkedHashMap<String,double[]>();
		try {
		    
		    att_num1=0;

		    in = new BufferedReader(new FileReader(file));

		    while (true)
			{
			    
			    String line = in.readLine(); // %% class Label
			    
			    //System.out.println(line);
			    
			    if(line==null)
				break;
			    String[] splitline = line.split(DELIMITER);
			    
			    if(!dates.equals(splitline[2]) && splitline[0].equals("1") && splitline[1].equals("1"))
			       {
				   dates=splitline[2];
				   total_train_dates++;
			       }
			    
			  
			    if(!splitline[1].equals("Dept"))
				if(depNum<Integer.parseInt(splitline[1]))
				    depNum=Integer.parseInt(splitline[1]);

			    if(att_num1==0)
				{
				    att_num1=splitline.length;
				    
				    max_val1 = new double[att_num1-1];
				    min_val1 = new double[att_num1-1];
				    
				    for (int i=0; i<att_num1; i++)
					{
					    if(i<2)
						{
						    max_val1[i]=0;
						    min_val1[i]=10000000;
						}
					    else if(i>2)
						{
						    max_val1[i-1]=0;
						    min_val1[i-1]=10000000;
						}
					}
			   
				}
			    else
				{
				    double f_val[] = new double[att_num1-1];
				    
				    String key = splitline[0] + "a"+ splitline[2]+"b"+splitline[1];
				    
				    for (int i=0; i<att_num1;i++)
					{
					    if(i<2)
						{
						    if(splitline[i].equals("NA") | splitline[i].equals("FALSE"))
						       f_val[i]= 0;
						    else if(splitline[i].equals("TRUE"))
						       f_val[i]= 1;
						    else
							f_val[i]=Double.parseDouble(splitline[i]);

						    if(max_val1[i]<f_val[i])
							max_val1[i]=f_val[i];
						    if(min_val1[i]>f_val[i])
							min_val1[i]=f_val[i];

						    
						}
					    else if(i>2)
						{
						    if(splitline[i].equals("NA") | splitline[i].equals("FALSE"))
						       f_val[i-1]= 0;
						    else if(splitline[i].equals("TRUE"))
						       f_val[i-1]= 1;
						    else
							f_val[i-1]=Double.parseDouble(splitline[i]);

						    if(max_val1[i-1]<f_val[i-1])
							max_val1[i-1]=f_val[i-1];
						    if(min_val1[i-1]>f_val[i-1])
							min_val1[i-1]=f_val[i-1];

						}
					}
						    
						       
					       
				    Train.put(key, f_val);
				}

			    
			}
			
			    in.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		} 

		
		System.out.println("Train Set Updated");
	      
		
		max_val1[2]=500000;
		min_val1[2]=-10000;
		for(String key: Train.keySet())
		    {
			
			double[] f_val = Train.get(key);
		       
			for (int i=0; i<att_num1-1; i++)
			    {
				if(Math.abs(max_val1[i]-min_val1[i])>0.0001)
				    f_val[i]=0 + 1.0*(f_val[i]-min_val1[i])/(max_val1[i]-min_val1[i]);
				else
				    f_val[i]=1.0;
				
			    }
			Train.put(key,f_val);
		    }
		
	      
	}
	public static void createTest(String file) {
      
		BufferedReader in;
		final String DELIMITER = ",";
		
		String dates="a";
		
		total_test_dates=0;
		depNum=0;
		
	        Test = new LinkedHashMap<String,double[]>();
		try {
		    
		    att_num1=0;

		    in = new BufferedReader(new FileReader(file));

		    while (true)
			{
			    
			    String line = in.readLine(); // %% class Label
			    
			    //System.out.println(line);
			    
			    if(line==null) break;
			    String[] splitline = line.split(DELIMITER);
			    
			    if(!dates.equals(splitline[2]) && splitline[0].equals("1") && splitline[1].equals("1"))
			       {
				   dates=splitline[2];
				   total_test_dates++;
			       }
			    
			  
			    if(!splitline[1].equals("Dept"))
				if(depNum<Integer.parseInt(splitline[1]))
				    depNum=Integer.parseInt(splitline[1]);

			    if(att_num1==0)
				{
				    att_num1=splitline.length;
				    
				    max_val1 = new double[att_num1-1];
				    min_val1 = new double[att_num1-1];
				    
				    for (int i=0; i<att_num1; i++)
					{
					    if(i<2)
						{
						    max_val1[i]=0;
						    min_val1[i]=10000000;
						}
					    else if(i>2)
						{
						    max_val1[i-1]=0;
						    min_val1[i-1]=10000000;
						}
					}
			   
				}
			    else
				{
				    double f_val[] = new double[att_num1-1];
				    
				    String key = splitline[0] + "a"+ splitline[2]+"b"+splitline[1];
				    
				    for (int i=0; i<att_num1;i++)
					{
					    if(i<2)
						{
						    if(splitline[i].equals("NA") | splitline[i].equals("FALSE"))
						       f_val[i]= 0;
						    else if(splitline[i].equals("TRUE"))
						       f_val[i]= 1;
						    else
							f_val[i]=Double.parseDouble(splitline[i]);

						    if(max_val1[i]<f_val[i])
							max_val1[i]=f_val[i];
						    if(min_val1[i]>f_val[i])
							min_val1[i]=f_val[i];

						    
						}
					    else if(i>2)
						{
						    if(splitline[i].equals("NA") | splitline[i].equals("FALSE"))
						       f_val[i-1]= 0;
						    else if(splitline[i].equals("TRUE"))
						       f_val[i-1]= 1;
						    else
							f_val[i-1]=Double.parseDouble(splitline[i]);

						    if(max_val1[i-1]<f_val[i-1])
							max_val1[i-1]=f_val[i-1];
						    if(min_val1[i-1]>f_val[i-1])
							min_val1[i-1]=f_val[i-1];
						}
					}
				    Test.put(key, f_val);
				}

			    
			}
			
			    in.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		} 

		
		System.out.println("Test Set Updated");
	      
		
		max_val1[2]=500000;
		min_val1[2]=-10000;
		for(String key: Test.keySet())
		    {
			
			double[] f_val = Test.get(key);
		       
			for (int i=0; i<att_num1-1; i++)
			    {
				if(Math.abs(max_val1[i]-min_val1[i])>0.0001)
				    f_val[i]=0 + 1.0*(f_val[i]-min_val1[i])/(max_val1[i]-min_val1[i]);
				else
				    f_val[i]=1.0;
				
			    }
			Test.put(key,f_val);
		    }
		
	
	      
	}
	
	
	
 }
