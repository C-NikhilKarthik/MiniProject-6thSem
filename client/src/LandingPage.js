import { useNavigation } from '@react-navigation/native';
import React from 'react'
import { StyleSheet, Button, ImageBackground, View, Text, StatusBar } from 'react-native'

const PatternBg = { uri: 'https://e1.pxfuel.com/desktop-wallpaper/759/194/desktop-wallpaper-subtle-pride-phone-background-made-by-me-r-lgbt-thumbnail.jpg' }

function LandingPage() {

  const navigation = useNavigation();

  return (
    <ImageBackground source={PatternBg} resizeMode='cover' className='h-full w-full  flex-none justify-center items-center rotate-180'>
      <View className="h-full w-full flex-none justify-center items-center rotate-180">
        <Text className="text-white text-5xl font-bold tracking-widest" >SNAPMARK</Text>
        <View className=" border-white border-b-4 w-9/12"></View>
        <Text className="text-white text-2xl" >Attendance in a Flash</Text>
        <Button
          color="rgb(101 163 13)"
          title="Continue"
          onPress={() => {
            navigation.navigate('homePage')
          }}
        />
        <StatusBar style="auto" />
      </View>
    </ImageBackground>
  )
}

export default LandingPage